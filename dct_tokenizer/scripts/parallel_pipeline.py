#!/usr/bin/env python3
"""
Fully parallel pipeline: download + tokenize concurrently.
Uses multiple processes for downloading and feeding to tokenizer.
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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import io

def download_worker(dataset_name, split, image_key, start_idx, end_idx, image_queue, hf_token):
    """Worker that downloads images and puts them in queue."""
    os.environ['HF_TOKEN'] = hf_token
    os.environ['HF_HOME'] = '/tmp/hf_cache'
    
    from datasets import load_dataset
    
    ds = load_dataset(dataset_name, split=split, streaming=True)
    
    batch = []
    current_idx = 0
    
    for item in ds:
        if current_idx < start_idx:
            current_idx += 1
            continue
        if current_idx >= end_idx:
            break
        
        try:
            img = item[image_key]
            if img.mode != 'RGB':
                img = img.convert('RGB')
            batch.append((current_idx, img))
            
            if len(batch) >= 100:  # Send in chunks
                image_queue.put(batch)
                batch = []
        except:
            pass
        
        current_idx += 1
    
    if batch:
        image_queue.put(batch)
    
    image_queue.put(None)  # Signal done

def tokenize_worker(image_queue, output_queue, tokenizer_path, preset, num_downloaders):
    """Worker that tokenizes images from queue."""
    done_count = 0
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        batch_id = 0
        
        while done_count < num_downloaders:
            try:
                item = image_queue.get(timeout=1)
            except Empty:
                continue
            
            if item is None:
                done_count += 1
                continue
            
            # item is list of (idx, img) tuples
            batch_dir = tmpdir / f"batch_{batch_id}"
            batch_dir.mkdir(exist_ok=True)
            
            for i, (idx, img) in enumerate(item):
                img.save(batch_dir / f"{i:06d}.jpg", quality=90)
            
            # Run tokenizer
            out_file = tmpdir / f"tokens_{batch_id}.bin"
            cmd = [tokenizer_path, preset, str(batch_dir), str(out_file)]
            result = subprocess.run(cmd, capture_output=True)
            
            if result.returncode == 0:
                with open(out_file, 'rb') as f:
                    tokens = f.read()
                output_queue.put((len(item), tokens))
            else:
                output_queue.put((0, None))
            
            # Cleanup
            for f in batch_dir.iterdir():
                f.unlink()
            batch_dir.rmdir()
            if out_file.exists():
                out_file.unlink()
            
            batch_id += 1
    
    output_queue.put(None)  # Signal done

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('output')
    parser.add_argument('--preset', default='small')
    parser.add_argument('--split', default='train')
    parser.add_argument('--max-images', type=int, default=100000)
    parser.add_argument('--num-downloaders', type=int, default=4)
    parser.add_argument('--tokenizer', default='/root/dct_tokenizer/build/tokenize_compact')
    parser.add_argument('--image-key', default='image')
    args = parser.parse_args()
    
    hf_token = os.environ.get('HF_TOKEN', '')
    
    print(f"Dataset: {args.dataset}")
    print(f"Max images: {args.max_images}")
    print(f"Downloaders: {args.num_downloaders}")
    print(f"Preset: {args.preset}")
    
    # Create queues
    image_queue = mp.Queue(maxsize=50)  # Buffer up to 50 batches
    output_queue = mp.Queue()
    
    # Calculate ranges for each downloader
    images_per_worker = args.max_images // args.num_downloaders
    
    # Start download workers
    downloaders = []
    for i in range(args.num_downloaders):
        start = i * images_per_worker
        end = start + images_per_worker if i < args.num_downloaders - 1 else args.max_images
        
        p = mp.Process(target=download_worker, args=(
            args.dataset, args.split, args.image_key,
            start, end, image_queue, hf_token
        ))
        p.start()
        downloaders.append(p)
    
    # Start tokenize worker
    tokenizer = mp.Process(target=tokenize_worker, args=(
        image_queue, output_queue, args.tokenizer, args.preset, args.num_downloaders
    ))
    tokenizer.start()
    
    # Write results
    start_time = time.time()
    processed = 0
    
    with open(args.output, 'wb') as f:
        while True:
            try:
                item = output_queue.get(timeout=1)
            except Empty:
                continue
            
            if item is None:
                break
            
            count, tokens = item
            if tokens:
                f.write(tokens)
                processed += count
            
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"\rProcessed: {processed} | {rate:.0f} img/s    ", end='', flush=True)
    
    # Wait for all processes
    for p in downloaders:
        p.join()
    tokenizer.join()
    
    total_time = time.time() - start_time
    file_size = os.path.getsize(args.output) / (1024 * 1024)
    
    print(f"\n\n=== Done ===")
    print(f"Processed: {processed}")
    print(f"Time: {total_time:.1f}s ({processed/total_time:.0f} img/s)")
    print(f"Output: {args.output} ({file_size:.1f} MB)")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
