#!/usr/bin/env python3
"""Tokenize CC12M WebDataset shards into fixed-length 5-level pyramid+residual tokens.

Levels: 8 -> 16 -> 32 -> 64 -> 128

Tokens per image:
  - base 8x8 RGB: 8*8*3 = 192
  - residual 16x16: 16*16*3 = 768
  - residual 32x32: 32*32*3 = 3072
  - residual 64x64: 64*64*3 = 12288
  - residual 128x128: 128*128*3 = 49152
  Total = 65472 tokens/image, plus separator token 65535.

Residual encoding:
  resid_q = clip(resid + 128, 0, 255)

Output:
  - <out_dir>/token_shards/00000.bin ...
  - optional <out_dir>/tokenized.0
  - <out_dir>/config.json

This is a practical "high detail" tokenizer that avoids the extreme size of going all the way to 256.
"""

from __future__ import annotations

import argparse
import json
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEP = np.uint16(65535)


class PyramidResidualTokenizer5:
    def __init__(self):
        self.sizes = [8, 16, 32, 64, 128]
        self.tokens_per_level = [s * s * 3 for s in self.sizes]
        self.tokens_per_image = sum(self.tokens_per_level)
        self.vocab_size = 256
        self._placeholder = np.full((self.tokens_per_image,), 128, dtype=np.uint16)

    def placeholder(self) -> np.ndarray:
        return self._placeholder

    @staticmethod
    def _to_rgb(img: Image.Image) -> Image.Image:
        return img.convert("RGB") if img.mode != "RGB" else img

    def tokenize_pil(self, img: Image.Image) -> np.ndarray:
        img = self._to_rgb(img)

        targets = [np.asarray(img.resize((s, s), Image.BILINEAR), dtype=np.int16) for s in self.sizes]

        base = targets[0].astype(np.uint8)
        residuals = []

        recon = base
        for i in range(1, len(self.sizes)):
            s = self.sizes[i]
            up = np.asarray(Image.fromarray(recon, mode="RGB").resize((s, s), Image.BILINEAR), dtype=np.int16)
            resid = targets[i] - up
            resid_q = np.clip(resid + 128, 0, 255).astype(np.uint8)
            residuals.append(resid_q)
            recon = np.clip(up + (resid_q.astype(np.int16) - 128), 0, 255).astype(np.uint8)

        t = np.empty((self.tokens_per_image,), dtype=np.uint16)
        off = 0
        # base
        n0 = self.tokens_per_level[0]
        t[off : off + n0] = base.reshape(-1).astype(np.uint16)
        off += n0
        # residuals
        for i, resid_q in enumerate(residuals, start=1):
            n = self.tokens_per_level[i]
            t[off : off + n] = resid_q.reshape(-1).astype(np.uint16)
            off += n
        return t


def tokenize_one_tar(tar_path: str, out_bin: str) -> Tuple[str, int, int]:
    tok = PyramidResidualTokenizer5()
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
    ap.add_argument("--out-dir", default="/root/cc12m_tokens_5lvl")
    ap.add_argument("--workers", type=int, default=32)
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
    tars = tars_all[start : (start + args.shard_count if args.shard_count and args.shard_count > 0 else None)]

    jobs = [(str(p), str(shard_out / (p.stem + ".bin"))) for p in tars]

    tok = PyramidResidualTokenizer5()
    print(f"Found {len(jobs)} tar shards in {shards_dir} (selected start={start}, count={args.shard_count or 'all'})")
    print(f"tokens_per_image={tok.tokens_per_image} vocab_size={tok.vocab_size} levels={tok.sizes}")
    print(f"Starting tokenization with {min(args.workers, len(jobs))} workers...")

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
        "tokenizer": "pyramid_residual_5lvl",
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
