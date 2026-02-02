#!/usr/bin/env python3
"""Check if a given image corresponds exactly to a training document.

We (re-)tokenize the image using the 4-level pyramid residual scheme (8/16/32/64)
and then query the shardwise Fastgram indices for an exact match of:
  tokens + [eos]

If count>0, the image token sequence exists verbatim in the training set.

Note: If the image was generated and then upsampled to 256x256, re-tokenization
may not reproduce the exact original tokens. For the most reliable check, use
--image-size 64 if the generated images were saved at 64x64.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def tokenize_4lvl(img: Image.Image) -> np.ndarray:
    # force RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    def rs(sz: int) -> np.ndarray:
        return np.asarray(img.resize((sz, sz), Image.BILINEAR), dtype=np.int16)

    a8 = rs(8)
    a16 = rs(16)
    a32 = rs(32)
    a64 = rs(64)

    up16 = np.asarray(Image.fromarray(a8.astype(np.uint8), mode="RGB").resize((16, 16), Image.BILINEAR), dtype=np.int16)
    r16 = np.clip((a16 - up16) + 128, 0, 255).astype(np.uint8)
    recon16 = np.clip(up16 + (r16.astype(np.int16) - 128), 0, 255).astype(np.uint8)

    up32 = np.asarray(Image.fromarray(recon16, mode="RGB").resize((32, 32), Image.BILINEAR), dtype=np.int16)
    r32 = np.clip((a32 - up32) + 128, 0, 255).astype(np.uint8)
    recon32 = np.clip(up32 + (r32.astype(np.int16) - 128), 0, 255).astype(np.uint8)

    up64 = np.asarray(Image.fromarray(recon32, mode="RGB").resize((64, 64), Image.BILINEAR), dtype=np.int16)
    r64 = np.clip((a64 - up64) + 128, 0, 255).astype(np.uint8)

    t = np.empty((16320,), dtype=np.uint16)
    o0, o1, o2 = 192, 192 + 768, 192 + 768 + 3072
    t[:o0] = a8.astype(np.uint8).reshape(-1).astype(np.uint16)
    t[o0:o1] = r16.reshape(-1).astype(np.uint16)
    t[o1:o2] = r32.reshape(-1).astype(np.uint16)
    t[o2:] = r64.reshape(-1).astype(np.uint16)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to image to check")
    ap.add_argument("--index-root", default="/root/cc12m_indices_4lvl")
    ap.add_argument("--config", default="/root/cc12m_tokens_4lvl/config.json")
    ap.add_argument("--group-size", type=int, default=10)
    ap.add_argument("--fanout-threads", type=int, default=0)
    ap.add_argument("--threads-per-engine", type=int, default=1)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    eos = int(cfg.get("separator_token", 65535))
    vocab = int(cfg.get("vocab_size", 256))

    from multishard_engine import MultiShardGramEngine

    engine = MultiShardGramEngine(
        index_root=args.index_root,
        eos_token_id=eos,
        vocab_size=vocab,
        group_size=args.group_size,
        fanout_threads=(args.fanout_threads if args.fanout_threads > 0 else None),
        threads_per_engine=args.threads_per_engine,
    )

    img = Image.open(args.image)
    toks = tokenize_4lvl(img)

    # Query exact doc match: tokens + eos
    query = toks.astype(int).tolist() + [eos]

    print(f"Query length (incl eos): {len(query)}")
    r = engine.count(query)
    print("count(tokens+eos):", r)

    # Also check without eos (can match across boundaries)
    r2 = engine.count(toks.astype(int).tolist())
    print("count(tokens):", r2)


if __name__ == "__main__":
    main()
