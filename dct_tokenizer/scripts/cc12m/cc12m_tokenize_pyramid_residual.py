#!/usr/bin/env python3
"""Tokenize CC12M WebDataset shards into fixed-length pyramid+residual tokens.

Input: a directory of WebDataset .tar shards (each ~10k images)
Output:
  - per-shard token files: <out_dir>/tokens_shards/00000.bin, ...
  - concatenated token stream: <out_dir>/tokenized.0
  - config.json

Tokenization (training-free, fast):
  - decode image to RGB
  - resize to 8x8 (base) and 16x16 (detail) directly from original
  - residual = level16 - upsample(level8->16)
  - tokens = [base8 pixels (0..255), residual16+128 clipped (0..255)]

Each image produces exactly 960 uint16 tokens + separator (65535).
Vocab size: 256 (token values 0..255).

Designed for high throughput on CPU by parallelizing across shards.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import tarfile
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEP = np.uint16(65535)


class PyramidResidualTokenizer:
    def __init__(self, base: int = 8, detail: int = 16):
        assert detail == base * 2, "This tokenizer assumes detail = base*2"
        self.base = base
        self.detail = detail
        self.tokens_per_image = (base * base * 3) + (detail * detail * 3)
        self.vocab_size = 256
        # placeholder: mid-gray base, zero residual
        self._placeholder = np.full((self.tokens_per_image,), 128, dtype=np.uint16)

    def placeholder(self) -> np.ndarray:
        return self._placeholder

    def tokenize_pil(self, img: Image.Image) -> np.ndarray:
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Downsample directly from original (avoid 256x256 resize)
        im8 = img.resize((self.base, self.base), Image.BILINEAR)
        im16 = img.resize((self.detail, self.detail), Image.BILINEAR)

        a8 = np.asarray(im8, dtype=np.int16)  # 0..255
        a16 = np.asarray(im16, dtype=np.int16)

        # Upsample 8->16
        up = Image.fromarray(a8.astype(np.uint8), mode="RGB").resize((self.detail, self.detail), Image.BILINEAR)
        up16 = np.asarray(up, dtype=np.int16)

        residual = a16 - up16  # roughly [-255,255]
        residual_q = np.clip(residual + 128, 0, 255).astype(np.uint8)

        tokens = np.empty((self.tokens_per_image,), dtype=np.uint16)
        # base
        tokens[: self.base * self.base * 3] = a8.astype(np.uint8).reshape(-1).astype(np.uint16)
        # residual
        tokens[self.base * self.base * 3 :] = residual_q.reshape(-1).astype(np.uint16)
        return tokens


def tokenize_one_tar(tar_path: str, out_bin: str) -> Tuple[str, int, int]:
    """Tokenize one .tar shard to a .bin token stream.

    Returns: (tar_name, images_written, images_failed)
    """
    tokenizer = PyramidResidualTokenizer(8, 16)
    images = 0
    failed = 0

    tar_path_p = Path(tar_path)
    out_tmp = Path(out_bin + ".tmp")
    out_final = Path(out_bin)
    out_tmp.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path_p, mode="r") as tf, open(out_tmp, "wb", buffering=1024 * 1024) as out:
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
                tokens = tokenizer.placeholder()
            else:
                try:
                    img = Image.open(f)
                    tokens = tokenizer.tokenize_pil(img)
                except Exception:
                    failed += 1
                    tokens = tokenizer.placeholder()

            out.write(tokens.tobytes())
            out.write(sep_bytes)
            images += 1

    out_tmp.replace(out_final)
    return tar_path_p.name, images, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", required=True, help="Directory containing CC12M .tar shards")
    ap.add_argument("--out-dir", default="/root/cc12m_tokens", help="Output directory")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--concat", action="store_true", help="Concatenate shard .bin files into tokenized.0")
    args = ap.parse_args()

    shards_dir = Path(args.shards_dir)
    out_dir = Path(args.out_dir)
    shard_out_dir = out_dir / "token_shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_out_dir.mkdir(parents=True, exist_ok=True)

    tar_paths = sorted(shards_dir.glob("*.tar"))
    if not tar_paths:
        raise SystemExit(f"No .tar files found in {shards_dir}")

    print(f"Found {len(tar_paths)} tar shards in {shards_dir}")

    # Map tar -> output bin
    jobs = []
    for p in tar_paths:
        out_bin = shard_out_dir / (p.stem + ".bin")
        jobs.append((str(p), str(out_bin)))

    # Tokenize in parallel across shards
    from concurrent.futures import ProcessPoolExecutor, as_completed

    start = time.time()
    done_images = 0
    done_failed = 0

    print(f"Starting tokenization with {args.workers} workers...")

    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
        futs = [ex.submit(tokenize_one_tar, tar, out) for tar, out in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            name, n_img, n_fail = fut.result()
            done_images += n_img
            done_failed += n_fail
            elapsed = time.time() - start
            rate = done_images / max(elapsed, 1e-9)
            print(
                f"[{i}/{len(jobs)}] {name}: {n_img} imgs ({n_fail} failed). "
                f"Total {done_images:,} imgs, {rate:,.0f} img/s"
            )

    total = time.time() - start
    print(f"\nDone tokenizing {done_images:,} images in {total:.1f}s ({done_images/total:,.0f} img/s)")

    tokenizer = PyramidResidualTokenizer(8, 16)
    config = {
        "tokenizer": "pyramid_residual",
        "encoded_levels": [8, 16],
        "tokens_per_image": tokenizer.tokens_per_image,
        "vocab_size": tokenizer.vocab_size,
        "separator_token": int(SEP),
        "num_images": int(done_images),
        "num_failed": int(done_failed),
        "source": str(shards_dir),
        "shards": len(tar_paths),
    }

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Optional concat (tokenized.0)
    if args.concat:
        out_tokenized = out_dir / "tokenized.0"
        print(f"Concatenating into {out_tokenized}...")
        with open(out_tokenized, "wb", buffering=1024 * 1024) as out:
            for p in sorted(shard_out_dir.glob("*.bin")):
                with open(p, "rb", buffering=1024 * 1024) as inp:
                    while True:
                        buf = inp.read(1024 * 1024)
                        if not buf:
                            break
                        out.write(buf)
        print("Concat done.")


if __name__ == "__main__":
    main()
