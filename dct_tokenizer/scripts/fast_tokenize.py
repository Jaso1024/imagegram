#!/usr/bin/env python3
"""
Fast image tokenization using in-memory processing.
Loads batches from HuggingFace, tokenizes with C++ tool, outputs tokens directly.
"""

import os
import sys
import argparse
import struct
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

def process_batch(batch_data, tokenizer_path, preset, tmpdir, batch_id):
    """Process a batch of images through the tokenizer."""
    batch_dir = Path(tmpdir) / f"batch_{batch_id}"
    batch_dir.mkdir(exist_ok=True)
    
    # Save images
    for i, img in enumerate(batch_data):
        img.save(batch_dir / f"{i:06d}.jpg", quality=95)
    
    # Run tokenizer
    out_file = Path(tmpdir) / f"tokens_{batch_id}.bin"
    cmd = [tokenizer_path, preset, str(batch_dir), str(out_file)]
    result = subprocess.run(cmd, capture_output=True)
    
    if result.returncode != 0:
        print(f"Tokenizer error: {result.stderr.decode()}", file=sys.stderr)
        return None
    
    # Read tokens
    with open(out_file, 'rb') as f:
        tokens = f.read()
    
    # Cleanup
    for f in batch_dir.iterdir():
        f.unlink()
    batch_dir.rmdir()
    out_file.unlink()
    
    return tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='HuggingFace dataset name')
    parser.add_argument('output', help='Output file')
    parser.add_argument('--preset', default='small', choices=['tiny', 'small', 'medium', 'large'])
    parser.add_argument('--split', default='train')
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--tokenizer', default='/root/dct_tokenizer/build/tokenize_compact')
    parser.add_argument('--image-key', default='image')
    parser.add_argument('--num-proc', type=int, default=4, help='Parallel dataset loading')
    args = parser.parse_args()
    
    os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN', '')
    
    from datasets import load_dataset
    
    print(f"Loading {args.dataset}...")
    ds = load_dataset(args.dataset, split=args.split)
    
    total = len(ds)
    if args.max_images:
        total = min(total, args.max_images)
        ds = ds.select(range(total))
    
    print(f"Processing {total} images with preset '{args.preset}'")
    
    with open(args.output, 'wb') as out_f:
        with tempfile.TemporaryDirectory() as tmpdir:
            start_time = time.time()
            processed = 0
            
            # Process in batches
            for batch_start in range(0, total, args.batch_size):
                batch_end = min(batch_start + args.batch_size, total)
                
                # Extract images from dataset
                batch_images = []
                for i in range(batch_start, batch_end):
                    img = ds[i][args.image_key]
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    batch_images.append(img)
                
                # Process batch
                tokens = process_batch(
                    batch_images,
                    args.tokenizer,
                    args.preset,
                    tmpdir,
                    batch_start
                )
                
                if tokens:
                    out_f.write(tokens)
                    processed += len(batch_images)
                
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                pct = 100 * batch_end / total
                eta = (total - batch_end) / rate if rate > 0 else 0
                
                print(f"\r[{pct:.1f}%] {processed}/{total} | {rate:.0f} img/s | ETA: {eta:.0f}s    ", end='', flush=True)
    
    total_time = time.time() - start_time
    file_size = os.path.getsize(args.output) / (1024 * 1024)
    
    print(f"\n\n=== Done ===")
    print(f"Processed: {processed} images")
    print(f"Time: {total_time:.1f}s ({processed/total_time:.0f} img/s)")
    print(f"Output: {args.output} ({file_size:.1f} MB)")

if __name__ == '__main__':
    main()
