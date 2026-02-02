#!/usr/bin/env python3
"""Generate images by n-gram sampling from CC12M shardwise Fastgram indices.

Uses MultiShardGramEngine (fan-out + merge across shard indices).
Token format (per image):
  - base 8x8 RGB pixels: 8*8*3 = 192 tokens in [0,255]
  - residual 16x16 RGB: 16*16*3 = 768 tokens in [0,255] representing residual+128
Total: 960 tokens.

Reconstruction:
  base8 -> upsample to 16x16 -> add (residual-128) -> clip -> upsample to output_size.

Example:
  python3 cc12m_generate.py \
    --index-root /root/cc12m_indices \
    --out-dir /root/cc12m_generated \
    --num-images 16
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

    # temperature
    if temperature != 1.0:
        logp = np.log(probs + 1e-12) / temperature
        probs = np.exp(logp)
        probs /= probs.sum()

    # top-k
    if 0 < top_k < len(probs):
        idx = np.argpartition(probs, -top_k)[-top_k:]
        mask = np.zeros_like(probs)
        mask[idx] = 1
        probs = probs * mask
        s = probs.sum()
        if s <= 0:
            probs = mask / mask.sum()
        else:
            probs /= s

    return int(np.random.choice(tokens, p=probs))


def generate_tokens(
    engine,
    num_tokens: int,
    temperature: float = 1.0,
    top_k: int = 50,
    max_context: int = 16,
) -> np.ndarray:
    # First token from unigram
    r = engine.ntd([], max_support=max(256, top_k))
    first = sample_from_ntd(r.get("result_by_token_id", {}), temperature=temperature, top_k=top_k)
    if first is None:
        first = 128

    seq: List[int] = [first]

    while len(seq) < num_tokens:
        nxt = None
        # backoff by context length
        for ctx_len in range(min(max_context, len(seq)), 0, -1):
            ctx = seq[-ctx_len:]
            r = engine.ntd(ctx, max_support=max(256, top_k))
            nxt = sample_from_ntd(r.get("result_by_token_id", {}), temperature=temperature, top_k=top_k)
            if nxt is not None:
                break

        if nxt is None:
            r = engine.ntd([], max_support=max(256, top_k))
            nxt = sample_from_ntd(r.get("result_by_token_id", {}), temperature=temperature, top_k=top_k)

        if nxt is None:
            nxt = 128

        seq.append(nxt)

    return np.array(seq[:num_tokens], dtype=np.uint16)


def decode_pyramid_residual(tokens: np.ndarray, output_size: int = 256) -> np.ndarray:
    assert tokens.shape[0] == 960
    base = tokens[:192].astype(np.uint8).reshape(8, 8, 3)
    resid_q = tokens[192:].astype(np.int16).reshape(16, 16, 3)

    # upsample base to 16x16
    up = Image.fromarray(base, mode="RGB").resize((16, 16), Image.BILINEAR)
    up16 = np.asarray(up, dtype=np.int16)

    resid = resid_q - 128
    img16 = np.clip(up16 + resid, 0, 255).astype(np.uint8)

    if output_size != 16:
        img = Image.fromarray(img16, mode="RGB").resize((output_size, output_size), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)
    return img16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", default="/root/cc12m_indices")
    ap.add_argument("--config", default="/root/cc12m_tokens/config.json")
    ap.add_argument("--out-dir", default="/root/cc12m_generated")
    ap.add_argument("--num-images", type=int, default=16)
    ap.add_argument("--output-size", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-context", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=10)
    ap.add_argument("--fanout-threads", type=int, default=0)
    ap.add_argument("--threads-per-engine", type=int, default=1)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    tokens_per_image = int(cfg["tokens_per_image"])
    assert tokens_per_image == 960, f"Expected 960 tokens/image, got {tokens_per_image}"

    # Import the multishard wrapper (expected to be available on server)
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

    # Save settings
    settings = {
        "index_root": args.index_root,
        "tokens_per_image": tokens_per_image,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_context": args.max_context,
        "group_size": args.group_size,
        "fanout_threads": args.fanout_threads,
        "threads_per_engine": args.threads_per_engine,
    }
    (out_dir / "gen_settings.json").write_text(json.dumps(settings, indent=2))

    for i in range(args.num_images):
        print(f"Generating {i+1}/{args.num_images}...")
        toks = generate_tokens(
            engine,
            num_tokens=tokens_per_image,
            temperature=args.temperature,
            top_k=args.top_k,
            max_context=args.max_context,
        )
        img = decode_pyramid_residual(toks, output_size=args.output_size)
        Image.fromarray(img).save(out_dir / f"gen_{i:04d}.png")

    print(f"Done. Wrote images to {out_dir}")


if __name__ == "__main__":
    main()
