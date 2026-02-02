#!/usr/bin/env python3
"""Tokenize CC12M WebDataset shards into a 6-level pyramid+residual token stream (training-free).

Why base=4 (not 8):
- A true 6-level starting from 8 would end at 256 (8,16,32,64,128,256) which is
  262,080 tokens/image (~524KB/image) -> ~524GB tokens for 1M images.
- base=4 gives 6 levels ending at 128 (4,8,16,32,64,128): 65,520 tokens/image
  (~131KB/image) -> ~131GB tokens for 1M images. Still big but feasible.

Levels encoded:
  L0: base 4x4 RGB (48 tokens)
  L1: residual 8x8   vs upsample(recon4->8)    (192 tokens, stored as residual+128 clipped to [0,255])
  L2: residual 16x16 vs upsample(recon8->16)   (768 tokens)
  L3: residual 32x32 vs upsample(recon16->32)  (3072 tokens)
  L4: residual 64x64 vs upsample(recon32->64)  (12288 tokens)
  L5: residual 128x128 vs upsample(recon64->128) (49152 tokens)

Total tokens/image = 65520, plus separator 65535.
Vocab size = 256 (0..255).

Outputs:
  - per-shard token files: <out_dir>/token_shards/00000.bin ...
  - optional concatenated stream: <out_dir>/tokenized.0
  - <out_dir>/config.json

Parallelizes across shards.

Recommended usage:
  # Start small (1 shard) to sanity check speed/size:
  python3 cc12m_tokenize_pyramid_residual_6lvl.py --shards-dir /root/cc12m_wds --out-dir /root/cc12m_tokens_6lvl --workers 16 --shard-start 0 --shard-count 1
"""

from __future__ import annotations

import argparse
import json
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEP = np.uint16(65535)


class PyramidResidualTokenizer6:
    def __init__(self, base: int = 4):
        self.sizes: List[int] = [base * (2**i) for i in range(6)]  # 4,8,16,32,64,128
        assert self.sizes[0] == 4 and self.sizes[-1] == 128, f"Unexpected sizes: {self.sizes}"
        self.tokens_per_level = [s * s * 3 for s in self.sizes]
        # tokens: base + all residuals (we store residuals at each level size)
        self.tokens_per_image = self.tokens_per_level[0] + sum(self.tokens_per_level[1:])
        self.vocab_size = 256
        self._placeholder = np.full((self.tokens_per_image,), 128, dtype=np.uint16)

    def placeholder(self) -> np.ndarray:
        return self._placeholder

    @staticmethod
    def _to_rgb(img: Image.Image) -> Image.Image:
        return img.convert("RGB") if img.mode != "RGB" else img

    def tokenize_pil(self, img: Image.Image) -> np.ndarray:
        img = self._to_rgb(img)

        # Downsample directly from original once per level
        targets = [np.asarray(img.resize((s, s), Image.BILINEAR), dtype=np.int16) for s in self.sizes]

        # Base
        base = targets[0].astype(np.uint8)

        # Progressive residuals
        residual_tokens: List[np.ndarray] = []

        recon = base
        for i in range(1, len(self.sizes)):
            s = self.sizes[i]
            up = np.asarray(Image.fromarray(recon, mode="RGB").resize((s, s), Image.BILINEAR), dtype=np.int16)
            resid = targets[i] - up
            resid_q = np.clip(resid + 128, 0, 255).astype(np.uint8)
            residual_tokens.append(resid_q)
            # update recon for next stage
            recon = np.clip(up + (resid_q.astype(np.int16) - 128), 0, 255).astype(np.uint8)

        # Pack tokens
        t = np.empty((self.tokens_per_image,), dtype=np.uint16)
        off = 0
        n0 = self.tokens_per_level[0]
        t[off : off + n0] = base.reshape(-1).astype(np.uint16)
        off += n0
        for i, resid_q in enumerate(residual_tokens, start=1):
            n = self.tokens_per_level[i]
            t[off : off + n] = resid_q.reshape(-1).astype(np.uint16)
            off += n
        return t


def tokenize_one_tar(tar_path: str, out_bin: str) -> Tuple[str, int, int]:
    tok = PyramidResidualTokenizer6(4)
    images = 0
    failed = 0

    tar_p = Path(tar_path)
    out_tmp = Path(out_bin + ".tmp")
    out_final = Path(out_bin)
    out_tmp.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_p, mode="r") as tf, open(out_tmp, "wb", buffering=1024 * 1024) as out:
        sep_bytes = SEP.tobytes()
        for member in tf:
            if not member.isfile():
                continue
            name = member.name
            if not (name.endswith(".jpg") or name.endswith(".jpeg") or name.endswith(".png")):
                continue

            f = tf.extractfile(member)
            if f is None:
                failed += 1
                tokens = tok.placeholder()
            else:
                try:
                    img = Image.open(f)
                    tokens = tok.tokenize_pil(img)
                except Exception:
                    failed += 1
                    tokens = tok.placeholder()

            out.write(tokens.tobytes())
            out.write(sep_bytes)
            images += 1

    out_tmp.replace(out_final)
    return tar_p.name, images, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", required=True)
    ap.add_argument("--out-dir", default="/root/cc12m_tokens_6lvl")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--concat", action="store_true")
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    shards_dir = Path(args.shards_dir)
    out_dir = Path(args.out_dir)
    shard_out = out_dir / "token_shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_out.mkdir(parents=True, exist_ok=True)

    tars_all = sorted(shards_dir.glob("*.tar"))
    if not tars_all:
        raise SystemExit(f"No .tar files in {shards_dir}")

    start = max(0, args.shard_start)
    if args.shard_count and args.shard_count > 0:
        tars = tars_all[start : start + args.shard_count]
    else:
        tars = tars_all[start:]

    jobs = [(str(p), str(shard_out / (p.stem + ".bin"))) for p in tars]

    print(f"Found {len(jobs)} tar shards in {shards_dir} (selected start={start}, count={args.shard_count or 'all'})")
    print(f"Starting tokenization with {min(args.workers, len(jobs))} workers...")

    tok = PyramidResidualTokenizer6(4)
    print(f"tokens_per_image={tok.tokens_per_image} vocab_size={tok.vocab_size} levels={tok.sizes}")

    t0 = time.time()
    done_images = 0
    done_failed = 0

    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
        futs = [ex.submit(tokenize_one_tar, tar, out) for tar, out in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            name, n_img, n_fail = fut.result()
            done_images += n_img
            done_failed += n_fail
            elapsed = time.time() - t0
            rate = done_images / max(elapsed, 1e-9)
            print(f"[{i}/{len(jobs)}] {name}: {n_img} imgs ({n_fail} failed). Total {done_images:,} imgs, {rate:,.0f} img/s")

    total = time.time() - t0
    print(f"\nDone tokenizing {done_images:,} images in {total:.1f}s ({done_images/total:,.0f} img/s)")

    cfg = {
        "tokenizer": "pyramid_residual_6lvl",
        "encoded_levels": tok.sizes,
        "tokens_per_image": tok.tokens_per_image,
        "vocab_size": tok.vocab_size,
        "separator_token": int(SEP),
        "num_images": int(done_images),
        "num_failed": int(done_failed),
        "source": str(shards_dir),
        "shards": len(tars),
        "shard_start": start,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    if args.concat:
        out_tokenized = out_dir / "tokenized.0"
        print(f"Concatenating into {out_tokenized}...")
        with open(out_tokenized, "wb", buffering=1024 * 1024) as out:
            for p in sorted(shard_out.glob("*.bin")):
                with open(p, "rb", buffering=1024 * 1024) as inp:
                    while True:
                        buf = inp.read(1024 * 1024)
                        if not buf:
                            break
                        out.write(buf)
        print("Concat done.")


if __name__ == "__main__":
    main()
