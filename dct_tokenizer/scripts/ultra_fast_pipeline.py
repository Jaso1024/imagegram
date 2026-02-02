#!/usr/bin/env python3
"""
Ultra-fast pipeline using hf_transfer for ~1GB/s downloads.
Downloads parquets -> Extracts images -> Tokenizes in parallel.
"""

import os
import sys
import time
import argparse
import pyarrow.parquet as pq
import io
import tempfile
import subprocess
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from multiprocessing import Pool, cpu_count
import glob

# Enable hf_transfer
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

def extract_images_from_parquet(args):
    """Extract images from a single parquet file."""
    parquet_path, output_dir, start_idx = args
    
    table = pq.read_table(parquet_path)
    count = 0
    
    for i in range(len(table)):
        try:
            img_data = table['image'][i].as_py()
            img_bytes = img_data['bytes']
            img = Image.open(io.BytesIO(img_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(f'{output_dir}/{start_idx + i:08d}.jpg', quality=90)
            count += 1
        except Exception as e:
            pass
    
    return count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--preset', default='small')
    parser.add_argument('--num-shards', type=int, default=40, help='Number of shards (max 40)')
    parser.add_argument('--tokenizer', default='/root/dct_tokenizer/build/tokenize_compact')
    parser.add_argument('--dataset', default='benjamin-paine/imagenet-1k-256x256')
    parser.add_argument('--extract-workers', type=int, default=8)
    args = parser.parse_args()
    
    # Authentication: set HF_TOKEN in your environment for gated datasets / higher rate limits.
    # Example: export HF_TOKEN=hf_... 
    if 'HF_TOKEN' not in os.environ:
        print('Warning: HF_TOKEN not set. If the dataset is gated or you hit rate limits, set HF_TOKEN.')
    os.environ['HF_HOME'] = '/workspace/.hf_home'
    
    from huggingface_hub import snapshot_download
    
    print(f'Dataset: {args.dataset}')
    print(f'Shards: {args.num_shards}')
    print(f'Preset: {args.preset}')
    print(f'Extract workers: {args.extract_workers}')
    print()
    
    total_start = time.time()
    
    # Stage 1: Download parquets with hf_transfer
    print('='*60)
    print('STAGE 1: Downloading parquets (hf_transfer)')
    print('='*60)
    
    # Build pattern for shards
    shard_patterns = [f'data/train-{i:05d}-of-00040.parquet' for i in range(args.num_shards)]
    pattern = 'data/train-000[0-3]?-of-00040.parquet' if args.num_shards == 40 else None
    
    download_start = time.time()
    
    if args.num_shards <= 10:
        # Use specific patterns for small number of shards
        patterns = [f'data/train-{i:05d}-of-00040.parquet' for i in range(args.num_shards)]
        dataset_path = snapshot_download(
            repo_id=args.dataset,
            repo_type='dataset',
            allow_patterns=patterns,
        )
    else:
        # Download all data
        dataset_path = snapshot_download(
            repo_id=args.dataset,
            repo_type='dataset',
            allow_patterns='data/*.parquet',
        )
    
    download_time = time.time() - download_start
    
    # Find downloaded parquets
    parquet_dir = Path(dataset_path) / 'data'
    parquet_files = sorted(glob.glob(str(parquet_dir / 'train-*.parquet')))[:args.num_shards]
    
    total_size = sum(os.path.getsize(f) for f in parquet_files)
    print(f'Downloaded {len(parquet_files)} files ({total_size/1e9:.1f} GB) in {download_time:.1f}s')
    print(f'Speed: {total_size/download_time/1e6:.0f} MB/s')
    
    # Stage 2: Extract images
    print()
    print('='*60)
    print('STAGE 2: Extracting images')
    print('='*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = Path(tmpdir) / 'images'
        img_dir.mkdir()
        
        extract_start = time.time()
        
        # Prepare extraction args
        extract_args = []
        start_idx = 0
        for pf in parquet_files:
            # Estimate ~32000 images per shard
            extract_args.append((pf, str(img_dir), start_idx))
            start_idx += 35000  # Leave some buffer
        
        # Extract in parallel using multiprocessing
        total_images = 0
        with Pool(processes=args.extract_workers) as pool:
            for i, count in enumerate(pool.imap(extract_images_from_parquet, extract_args)):
                total_images += count
                elapsed = time.time() - extract_start
                rate = total_images / elapsed if elapsed > 0 else 0
                print(f'\r  Shard {i+1}/{len(parquet_files)}: {total_images} images ({rate:.0f} img/s)', end='', flush=True)
        
        extract_time = time.time() - extract_start
        print(f'\nExtracted {total_images} images in {extract_time:.1f}s ({total_images/extract_time:.0f} img/s)')
        
        # Stage 3: Tokenize
        print()
        print('='*60)
        print('STAGE 3: Tokenizing')
        print('='*60)
        
        token_start = time.time()
        
        cmd = [args.tokenizer, args.preset, str(img_dir), args.output]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        token_time = time.time() - token_start
        
        if result.returncode == 0:
            file_size = os.path.getsize(args.output) / (1024 * 1024)
            print(f'Tokenized in {token_time:.1f}s ({total_images/token_time:.0f} img/s)')
            print(result.stdout)
        else:
            print(f'Error: {result.stderr}')
            return 1
    
    total_time = time.time() - total_start
    
    print()
    print('='*60)
    print('COMPLETE')
    print('='*60)
    print(f'Total images: {total_images}')
    print(f'Total time: {total_time:.1f}s')
    print(f'Effective rate: {total_images/total_time:.0f} img/s')
    print(f'Output: {args.output} ({file_size:.1f} MB)')
    print()
    print('Breakdown:')
    print(f'  Download: {download_time:6.1f}s ({100*download_time/total_time:4.1f}%) - {total_size/download_time/1e6:.0f} MB/s')
    print(f'  Extract:  {extract_time:6.1f}s ({100*extract_time/total_time:4.1f}%) - {total_images/extract_time:.0f} img/s')
    print(f'  Tokenize: {token_time:6.1f}s ({100*token_time/total_time:4.1f}%) - {total_images/token_time:.0f} img/s')

if __name__ == '__main__':
    main()
