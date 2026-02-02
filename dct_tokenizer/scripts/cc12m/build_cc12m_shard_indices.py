#!/usr/bin/env python3
"""Build Fastgram-compatible suffix-array indices for CC12M token shards.

We keep indices shardwise: one directory per shard containing:
  - table.0  (encoded suffix array)
  - tokenized.0 (copied from shard or symlink)
  - offset.0 (int64 token offsets per image)

Input token shard files are produced by cc12m_tokenize_pyramid_residual.py:
  /root/cc12m_tokens/token_shards/00000.bin ... 00099.bin

Each .bin is uint16 tokens stream with separator token 65535 after each image.
We infer num_images from file size: bytes_per_image = (tokens_per_image+1)*2.

This script runs fast_build_index on each shard in parallel.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Default values; can be overridden via --tokens-per-image.
TOKENS_PER_IMAGE_DEFAULT = 960



def build_one(
    bin_path: str,
    out_root: str,
    tokens_per_image: int,
    fast_build_index: str,
    token_width: int,
    keep_copy: bool = False,
) -> tuple[str, float]:
    bin_p = Path(bin_path)
    shard = bin_p.stem  # 00000
    out_dir = Path(out_root) / shard
    out_dir.mkdir(parents=True, exist_ok=True)

    table_path = out_dir / "table.0"
    if table_path.exists() and table_path.stat().st_size > 0:
        return shard, 0.0

    # temp input dir with tokenized.0
    tmp_in = out_dir / "_in"
    tmp_in.mkdir(exist_ok=True)
    tokenized_path = tmp_in / "tokenized.0"
    if tokenized_path.exists():
        tokenized_path.unlink()
    os.symlink(bin_p, tokenized_path)

    t0 = time.time()
    # Build suffix array table
    cmd = [fast_build_index, str(tmp_in), str(out_dir), str(token_width)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # offset file
    # NOTE: Fastgram expects offset.0 to be u64 values.
    # We store offsets in *token units* (not bytes) assuming token_width is known.
    # This is sufficient for count/ntd; doc-retrieval correctness depends on the exact
    # Fastgram convention (see Fastgram src/engine.cc for details).
    sep_tokens = tokens_per_image + 1
    bytes_per_image = sep_tokens * token_width
    size = bin_p.stat().st_size
    num_images = size // bytes_per_image
    offsets = np.arange(0, (num_images + 1) * sep_tokens, sep_tokens, dtype=np.uint64)
    (out_dir / "offset.0").write_bytes(offsets.tobytes())

    # Replace copied tokenized.0 with symlink to save space unless keep_copy
    if not keep_copy:
        tok_out = out_dir / "tokenized.0"
        if tok_out.exists() and not tok_out.is_symlink():
            tok_out.unlink()
            os.symlink(bin_p, tok_out)

    # cleanup temp
    try:
        tokenized_path.unlink()
        tmp_in.rmdir()
    except Exception:
        pass

    return shard, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-shards", required=True, help="Directory with *.bin token shards")
    ap.add_argument("--out-root", required=True, help="Output root dir for shard indices")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tokens-per-image", type=int, default=TOKENS_PER_IMAGE_DEFAULT)
    ap.add_argument("--token-width", type=int, default=2, help="Token width in bytes (u16=2)")
    ap.add_argument(
        "--fast-build-index",
        type=str,
        default="fast_build_index",
        help="Path to fast_build_index binary (default: fast_build_index in PATH)",
    )
    ap.add_argument("--keep-copy", action="store_true", help="Keep copied tokenized.0 in index dirs")
    args = ap.parse_args()

    token_dir = Path(args.token_shards)
    bins = sorted(token_dir.glob("*.bin"))
    if not bins:
        raise SystemExit(f"No .bin shards found in {token_dir}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Building indices for {len(bins)} shards -> {out_root}")
    start = time.time()

    done = 0
    total_time = 0.0

    # Resolve fast_build_index path. If default isn't found, but /root/fast_build_index exists (common on servers), use it.
    fbi = args.fast_build_index
    if fbi == "fast_build_index" and not shutil.which("fast_build_index") and Path("/root/fast_build_index").exists():
        fbi = "/root/fast_build_index"

    with ProcessPoolExecutor(max_workers=min(args.workers, len(bins))) as ex:
        futs = [
            ex.submit(
                build_one,
                str(p),
                str(out_root),
                int(args.tokens_per_image),
                str(fbi),
                int(args.token_width),
                args.keep_copy,
            )
            for p in bins
        ]
        for fut in as_completed(futs):
            shard, t = fut.result()
            done += 1
            if t > 0:
                total_time += t
            elapsed = time.time() - start
            rate = done / max(elapsed, 1e-9)
            print(f"[{done}/{len(bins)}] shard {shard} done in {t:.1f}s  ({rate:.2f} shards/s)")

    print(f"All shards complete in {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
