#!/usr/bin/env python3
"""Generate images by n-gram sampling from CC12M 3-level pyramid residual tokens.

Token format (per image):
  base 8x8 RGB: 192 tokens [0..255]
  resid16 (vs up8->16): 768 tokens [0..255] (residual+128)
  resid32 (vs recon16->32): 3072 tokens [0..255] (residual+128)
Total: 4032 tokens.

Decode:
  base8 -> up16 + (resid16-128) -> recon16
  recon16 -> up32 + (resid32-128) -> recon32
  recon32 -> upsample to output_size.

Uses MultiShardGramEngine to fan-out across shard indices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image


def sample_from_ntd(result_by_token_id: dict, temperature: float = 1.0, top_k: int = 50) -> Optional[int]:
    if not result_by_token_id:
        return None
    items = list(result_by_token_id.items())
    tokens = np.array([int(t) for t, _ in items], dtype=np.int32)
    probs = np.array([float(d["prob"]) for _, d in items], dtype=np.float64)

    if temperature != 1.0:
        logp = np.log(probs + 1e-12) / temperature
        probs = np.exp(logp)
        probs /= probs.sum()

    if 0 < top_k < len(probs):
        idx = np.argpartition(probs, -top_k)[-top_k:]
        mask = np.zeros_like(probs)
        mask[idx] = 1
        probs = probs * mask
        s = probs.sum()
        probs = probs / s if s > 0 else mask / mask.sum()

    return int(np.random.choice(tokens, p=probs))


def generate_tokens(engine, num_tokens: int, temperature: float, top_k: int, max_context: int) -> np.ndarray:
    r = engine.ntd([], max_support=max(1000, top_k))
    first = sample_from_ntd(r.get("result_by_token_id", {}), temperature=temperature, top_k=top_k)
    if first is None:
        first = 128

    seq: List[int] = [first]
    while len(seq) < num_tokens:
        nxt = None
        for ctx_len in range(min(max_context, len(seq)), 0, -1):
            ctx = seq[-ctx_len:]
            r = engine.ntd(ctx, max_support=max(1000, top_k))
            nxt = sample_from_ntd(r.get("result_by_token_id", {}), temperature=temperature, top_k=top_k)
            if nxt is not None:
                break
        if nxt is None:
            r = engine.ntd([], max_support=max(1000, top_k))
            nxt = sample_from_ntd(r.get("result_by_token_id", {}), temperature=temperature, top_k=top_k)
        if nxt is None:
            nxt = 128
        seq.append(nxt)

    return np.array(seq[:num_tokens], dtype=np.uint16)


def decode_3lvl(tokens: np.ndarray, output_size: int = 256) -> np.ndarray:
    assert tokens.shape[0] == 4032
    base = tokens[:192].astype(np.uint8).reshape(8, 8, 3)
    resid16_q = tokens[192:192 + 768].astype(np.int16).reshape(16, 16, 3)
    resid32_q = tokens[192 + 768:].astype(np.int16).reshape(32, 32, 3)

    up16 = np.asarray(Image.fromarray(base, mode="RGB").resize((16, 16), Image.BILINEAR), dtype=np.int16)
    recon16 = np.clip(up16 + (resid16_q - 128), 0, 255).astype(np.uint8)

    up32 = np.asarray(Image.fromarray(recon16, mode="RGB").resize((32, 32), Image.BILINEAR), dtype=np.int16)
    recon32 = np.clip(up32 + (resid32_q - 128), 0, 255).astype(np.uint8)

    if output_size != 32:
        out = Image.fromarray(recon32, mode="RGB").resize((output_size, output_size), Image.BILINEAR)
        return np.asarray(out, dtype=np.uint8)
    return recon32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", default="/root/cc12m_indices_3lvl")
    ap.add_argument("--config", default="/root/cc12m_tokens_3lvl/config.json")
    ap.add_argument("--out-dir", default="/root/cc12m_generated_3lvl")
    ap.add_argument("--num-images", type=int, default=16)
    ap.add_argument("--output-size", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-context", type=int, default=64)
    ap.add_argument("--group-size", type=int, default=10)
    ap.add_argument("--fanout-threads", type=int, default=0)
    ap.add_argument("--threads-per-engine", type=int, default=1)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    tpi = int(cfg["tokens_per_image"])
    assert tpi == 4032, f"Expected 4032 tokens/image, got {tpi}"

    from multishard_engine import MultiShardGramEngine

    engine = MultiShardGramEngine(
        index_root=args.index_root,
        eos_token_id=int(cfg.get("separator_token", 65535)),
        vocab_size=int(cfg.get("vocab_size", 256)),
        group_size=args.group_size,
        fanout_threads=(args.fanout_threads if args.fanout_threads > 0 else None),
        threads_per_engine=args.threads_per_engine,
        max_support=max(1000, args.top_k),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gen_settings.json").write_text(
        json.dumps(
            {
                "index_root": args.index_root,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "max_context": args.max_context,
                "group_size": args.group_size,
                "threads_per_engine": args.threads_per_engine,
            },
            indent=2,
        )
    )

    for i in range(args.num_images):
        print(f"Generating {i+1}/{args.num_images}...")
        toks = generate_tokens(engine, tpi, args.temperature, args.top_k, args.max_context)
        img = decode_3lvl(toks, output_size=args.output_size)
        Image.fromarray(img).save(out_dir / f"gen_{i:04d}.png")

    print(f"Done. Wrote images to {out_dir}")


if __name__ == "__main__":
    main()
