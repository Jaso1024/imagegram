#!/usr/bin/env python3
"""
Streaming tokenization - doesn't cache the full dataset to disk.
Uses streaming mode + multiprocessing for speed.
"""

import os
import sys
import argparse
import tempfile
import subprocess
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import time
import io

def writer_process(output_file, result_queue, done_event, total_expected):
    """Process that writes token results to file."""
    with open(output_file, 'wb') as f:
        written = 0
        while not (done_event.is_set() and result_queue.empty()):
            try:
                tokens = result_queue.get(timeout=0.1)
                if tokens:
                    f.write(tokens)
                    written += 1
            except Empty:
                continue
    return written

def tokenize_batch(images, tokenizer_path, preset, tmpdir):
    """Tokenize a batch of PIL images."""
    batch_dir = Path(tmpdir)
    batch_dir.mkdir(exist_ok=True)
    
    # Save images to temp dir
    for i, img in enumerate(images):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(batch_dir / f"{i:06d}.jpg", quality=90)
    
    # Run tokenizer
    out_file = batch_dir / "tokens.bin"
    cmd = [tokenizer_path, preset, str(batch_dir), str(out_file)]
    result = subprocess.run(cmd, capture_output=True)
    
    if result.returncode != 0:
        return None
    
    # Read and return tokens
    with open(out_file, 'rb') as f:
        tokens = f.read()
    
    # Cleanup
    for f_path in batch_dir.iterdir():
        f_path.unlink()
    
    return tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='HuggingFace dataset name')
    parser.add_argument('output', help='Output file')
    parser.add_argument('--preset', default='small')
    parser.add_argument('--split', default='train')
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--tokenizer', default='/root/dct_tokenizer/build/tokenize_compact')
    parser.add_argument('--image-key', default='image')
    args = parser.parse_args()
    
    os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN', '')
    os.environ['HF_HOME'] = '/tmp/hf_cache'  # Use temp for cache
    
    from datasets import load_dataset
    
    print(f"Loading {args.dataset} (streaming)...")
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    
    print(f"Tokenizing with preset '{args.preset}', batch_size={args.batch_size}")
    
    start_time = time.time()
    processed = 0
    failed = 0
    
    with open(args.output, 'wb') as out_f:
        batch = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for item in ds:
                if args.max_images and processed >= args.max_images:
                    break
                
                try:
                    img = item[args.image_key]
                    batch.append(img)
                except Exception as e:
                    failed += 1
                    continue
                
                if len(batch) >= args.batch_size:
                    # Process batch
                    tokens = tokenize_batch(batch, args.tokenizer, args.preset, tmpdir)
                    if tokens:
                        out_f.write(tokens)
                        processed += len(batch)
                    else:
                        failed += len(batch)
                    
                    batch = []
                    
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    print(f"\rProcessed: {processed} | Failed: {failed} | {rate:.0f} img/s    ", end='', flush=True)
            
            # Final batch
            if batch:
                tokens = tokenize_batch(batch, args.tokenizer, args.preset, tmpdir)
                if tokens:
                    out_f.write(tokens)
                    processed += len(batch)
    
    total_time = time.time() - start_time
    file_size = os.path.getsize(args.output) / (1024 * 1024)
    
    print(f"\n\n=== Done ===")
    print(f"Processed: {processed}")
    print(f"Failed: {failed}")
    print(f"Time: {total_time:.1f}s ({processed/total_time:.0f} img/s)")
    print(f"Output: {args.output} ({file_size:.1f} MB)")

if __name__ == '__main__':
    main()
