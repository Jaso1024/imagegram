#!/usr/bin/env python3
"""Tokenize CC12M WebDataset shards into fixed-length 3-level pyramid+residual tokens.

Levels:
  L0: base 8x8 RGB (192 tokens)
  L1: residual 16x16 RGB vs upsample(L0->16) (768 tokens, stored as residual+128 clipped to [0,255])
  L2: residual 32x32 RGB vs upsample(recon16->32) (3072 tokens, stored as residual+128 clipped to [0,255])

Total tokens/image = 192+768+3072 = 4032, plus separator 65535.
Vocab size = 256 (0..255).

Output:
  - per-shard token files: <out_dir>/token_shards/00000.bin ...
  - optional concatenated stream: <out_dir>/tokenized.0
  - <out_dir>/config.json

Designed for high throughput by parallelizing across shards.
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


class PyramidResidualTokenizer3:
    def __init__(self, base: int = 8):
        self.base = base
        self.l1 = base * 2
        self.l2 = base * 4
        self.tokens_per_image = (base * base * 3) + (self.l1 * self.l1 * 3) + (self.l2 * self.l2 * 3)
        self.vocab_size = 256
        self._placeholder = np.full((self.tokens_per_image,), 128, dtype=np.uint16)

    def placeholder(self) -> np.ndarray:
        return self._placeholder

    @staticmethod
    def _to_rgb(img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            return img.convert("RGB")
        return img

    def tokenize_pil(self, img: Image.Image) -> np.ndarray:
        img = self._to_rgb(img)

        # Downsample directly from original
        im8 = img.resize((self.base, self.base), Image.BILINEAR)
        im16 = img.resize((self.l1, self.l1), Image.BILINEAR)
        im32 = img.resize((self.l2, self.l2), Image.BILINEAR)

        a8 = np.asarray(im8, dtype=np.int16)
        a16 = np.asarray(im16, dtype=np.int16)
        a32 = np.asarray(im32, dtype=np.int16)

        # upsample 8 -> 16
        up16 = np.asarray(Image.fromarray(a8.astype(np.uint8), mode="RGB").resize((self.l1, self.l1), Image.BILINEAR), dtype=np.int16)
        resid16 = a16 - up16
        resid16_q = np.clip(resid16 + 128, 0, 255).astype(np.uint8)
        recon16 = np.clip(up16 + (resid16_q.astype(np.int16) - 128), 0, 255).astype(np.uint8)

        # upsample recon16 -> 32
        up32 = np.asarray(Image.fromarray(recon16, mode="RGB").resize((self.l2, self.l2), Image.BILINEAR), dtype=np.int16)
        resid32 = a32 - up32
        resid32_q = np.clip(resid32 + 128, 0, 255).astype(np.uint8)

        t = np.empty((self.tokens_per_image,), dtype=np.uint16)
        o0 = self.base * self.base * 3
        o1 = o0 + self.l1 * self.l1 * 3

        t[:o0] = a8.astype(np.uint8).reshape(-1).astype(np.uint16)
        t[o0:o1] = resid16_q.reshape(-1).astype(np.uint16)
        t[o1:] = resid32_q.reshape(-1).astype(np.uint16)
        return t


def tokenize_one_tar(tar_path: str, out_bin: str) -> Tuple[str, int, int]:
    tok = PyramidResidualTokenizer3(8)
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
    ap.add_argument("--out-dir", default="/root/cc12m_tokens_3lvl")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--concat", action="store_true")
    args = ap.parse_args()

    shards_dir = Path(args.shards_dir)
    out_dir = Path(args.out_dir)
    shard_out = out_dir / "token_shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_out.mkdir(parents=True, exist_ok=True)

    tars = sorted(shards_dir.glob("*.tar"))
    if not tars:
        raise SystemExit(f"No .tar files in {shards_dir}")

    jobs = [(str(p), str(shard_out / (p.stem + ".bin"))) for p in tars]

    print(f"Found {len(jobs)} tar shards in {shards_dir}")
    print(f"Starting tokenization with {min(args.workers, len(jobs))} workers...")

    start = time.time()
    done_images = 0
    done_failed = 0

    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
        futs = [ex.submit(tokenize_one_tar, tar, out) for tar, out in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            name, n_img, n_fail = fut.result()
            done_images += n_img
            done_failed += n_fail
            elapsed = time.time() - start
            rate = done_images / max(elapsed, 1e-9)
            print(f"[{i}/{len(jobs)}] {name}: {n_img} imgs ({n_fail} failed). Total {done_images:,} imgs, {rate:,.0f} img/s")

    total = time.time() - start
    print(f"\nDone tokenizing {done_images:,} images in {total:.1f}s ({done_images/total:,.0f} img/s)")

    tok = PyramidResidualTokenizer3(8)
    cfg = {
        "tokenizer": "pyramid_residual_3lvl",
        "encoded_levels": [8, 16, 32],
        "tokens_per_image": tok.tokens_per_image,
        "vocab_size": tok.vocab_size,
        "separator_token": int(SEP),
        "num_images": int(done_images),
        "num_failed": int(done_failed),
        "source": str(shards_dir),
        "shards": len(tars),
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
