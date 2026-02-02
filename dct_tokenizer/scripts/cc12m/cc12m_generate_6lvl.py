#!/usr/bin/env python3
"""Generate images by n-gram sampling from CC12M 6-level pyramid residual tokens.

This uses a 6-level scheme with base=4 and max encoded resolution 128:
Levels: 4,8,16,32,64,128
Tokens:
  base 4x4 RGB (48)
  residual 8x8 (192)
  residual 16x16 (768)
  residual 32x32 (3072)
  residual 64x64 (12288)
  residual 128x128 (49152)
Total tokens/image = 65520.

Decode:
  base4 -> up8 + (r8-128) -> recon8
  recon8 -> up16 + (r16-128) -> recon16
  recon16 -> up32 + (r32-128) -> recon32
  recon32 -> up64 + (r64-128) -> recon64
  recon64 -> up128 + (r128-128) -> recon128
  recon128 -> upsample to output_size (default 256)

WARNING:
- Generation is extremely slow because sequences are 65k tokens long.
- Start with --num-images 1 and consider smaller --max-context.
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


def decode_6lvl(tokens: np.ndarray, output_size: int = 256) -> np.ndarray:
    assert tokens.shape[0] == 65520

    sizes = [4, 8, 16, 32, 64, 128]
    counts = [s * s * 3 for s in sizes]
    offs = [0]
    for c in counts:
        offs.append(offs[-1] + c)

    base = tokens[offs[0] : offs[1]].astype(np.uint8).reshape(4, 4, 3)
    r8 = tokens[offs[1] : offs[2]].astype(np.int16).reshape(8, 8, 3)
    r16 = tokens[offs[2] : offs[3]].astype(np.int16).reshape(16, 16, 3)
    r32 = tokens[offs[3] : offs[4]].astype(np.int16).reshape(32, 32, 3)
    r64 = tokens[offs[4] : offs[5]].astype(np.int16).reshape(64, 64, 3)
    r128 = tokens[offs[5] : offs[6]].astype(np.int16).reshape(128, 128, 3)

    def up(img_u8: np.ndarray, sz: int) -> np.ndarray:
        return np.asarray(Image.fromarray(img_u8, mode="RGB").resize((sz, sz), Image.BILINEAR), dtype=np.int16)

    up8 = up(base, 8)
    recon8 = np.clip(up8 + (r8 - 128), 0, 255).astype(np.uint8)

    up16 = up(recon8, 16)
    recon16 = np.clip(up16 + (r16 - 128), 0, 255).astype(np.uint8)

    up32 = up(recon16, 32)
    recon32 = np.clip(up32 + (r32 - 128), 0, 255).astype(np.uint8)

    up64 = up(recon32, 64)
    recon64 = np.clip(up64 + (r64 - 128), 0, 255).astype(np.uint8)

    up128 = up(recon64, 128)
    recon128 = np.clip(up128 + (r128 - 128), 0, 255).astype(np.uint8)

    if output_size != 128:
        out = Image.fromarray(recon128, mode="RGB").resize((output_size, output_size), Image.BILINEAR)
        return np.asarray(out, dtype=np.uint8)
    return recon128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", default="/root/cc12m_indices_6lvl")
    ap.add_argument("--config", default="/root/cc12m_tokens_6lvl/config.json")
    ap.add_argument("--out-dir", default="/root/cc12m_generated_6lvl")
    ap.add_argument("--num-images", type=int, default=1)
    ap.add_argument("--output-size", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-context", type=int, default=32)
    ap.add_argument("--group-size", type=int, default=10)
    ap.add_argument("--fanout-threads", type=int, default=0)
    ap.add_argument("--threads-per-engine", type=int, default=1)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    tpi = int(cfg["tokens_per_image"])
    assert tpi == 65520, f"Expected 65520 tokens/image, got {tpi}"

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

    for i in range(args.num_images):
        print(f"Generating {i+1}/{args.num_images}...")
        toks = generate_tokens(engine, tpi, args.temperature, args.top_k, args.max_context)
        img = decode_6lvl(toks, output_size=args.output_size)
        Image.fromarray(img).save(out_dir / f"gen_{i:04d}.png")

    print(f"Done. Wrote images to {out_dir}")


if __name__ == "__main__":
    main()
