#!/usr/bin/env python3
"""
Fast end-to-end pipeline: Download parquets -> Extract images -> Tokenize
Achieves 1000+ img/s by parallelizing all stages.
"""

import os
import sys
import time
import argparse
import requests
import pyarrow.parquet as pq
import io
import tempfile
import subprocess
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Thread
import struct

def download_parquet(args):
    """Download a single parquet shard."""
    shard_id, base_url, headers, num_shards = args
    url = f'{base_url}/train-{shard_id:05d}-of-{num_shards:05d}.parquet'
    try:
        r = requests.get(url, headers=headers, timeout=300)
        r.raise_for_status()
        return shard_id, r.content
    except Exception as e:
        print(f'Error downloading shard {shard_id}: {e}', file=sys.stderr)
        return shard_id, None

def extract_single_image(args):
    """Extract and save a single image."""
    img_data, output_path = args
    try:
        img_bytes = img_data['bytes']
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(output_path, quality=90)
        return 1
    except:
        return 0

def extract_and_save_images(parquet_bytes, output_dir, start_idx, num_workers=64):
    """Extract images from parquet and save to disk (parallel)."""
    table = pq.read_table(io.BytesIO(parquet_bytes))
    
    # Prepare args for parallel processing
    args_list = []
    for i in range(len(table)):
        img_data = table['image'][i].as_py()
        output_path = output_dir / f'{start_idx + i:08d}.jpg'
        args_list.append((img_data, output_path))
    
    # Process in parallel with more workers
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(extract_single_image, args_list))
    
    return sum(results)

def tokenize_directory(tokenizer_path, preset, input_dir, output_file):
    """Run C++ tokenizer on directory."""
    cmd = [tokenizer_path, preset, str(input_dir), str(output_file)]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', required=True, help='Output token file')
    parser.add_argument('--preset', default='small', choices=['tiny', 'small', 'medium', 'large'])
    parser.add_argument('--num-shards', type=int, default=4, help='Number of parquet shards to download')
    parser.add_argument('--parallel-downloads', type=int, default=4)
    parser.add_argument('--tokenizer', default='/root/dct_tokenizer/build/tokenize_compact')
    parser.add_argument('--dataset', default='benjamin-paine/imagenet-1k-256x256')
    args = parser.parse_args()
    
    hf_token = os.environ.get('HF_TOKEN', '')
    headers = {'Authorization': f'Bearer {hf_token}'} if hf_token else {}
    
    # Determine number of shards in dataset
    num_total_shards = 40  # imagenet-1k-256x256 has 40 shards
    base_url = f'https://huggingface.co/datasets/{args.dataset}/resolve/main/data'
    
    print(f'Dataset: {args.dataset}')
    print(f'Downloading {args.num_shards} shards with {args.parallel_downloads} parallel connections')
    print(f'Tokenizer preset: {args.preset}')
    print()
    
    total_start = time.time()
    
    # Stage 1: Download parquets
    print('Stage 1: Downloading parquet files...')
    download_start = time.time()
    
    download_args = [(i, base_url, headers, num_total_shards) for i in range(args.num_shards)]
    
    with ThreadPoolExecutor(max_workers=args.parallel_downloads) as ex:
        results = list(ex.map(download_parquet, download_args))
    
    parquet_data = {shard_id: data for shard_id, data in results if data is not None}
    
    download_time = time.time() - download_start
    total_bytes = sum(len(d) for d in parquet_data.values())
    print(f'  Downloaded {total_bytes/1e6:.0f} MB in {download_time:.1f}s ({total_bytes/download_time/1e6:.0f} MB/s)')
    
    # Stage 2: Extract images to temp directory
    print('\nStage 2: Extracting images...')
    extract_start = time.time()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        img_dir = tmpdir / 'images'
        img_dir.mkdir()
        
        total_images = 0
        start_idx = 0
        
        for shard_id in sorted(parquet_data.keys()):
            data = parquet_data[shard_id]
            count = extract_and_save_images(data, img_dir, start_idx)
            total_images += count
            start_idx += count
            print(f'  Shard {shard_id}: {count} images')
        
        extract_time = time.time() - extract_start
        print(f'  Extracted {total_images} images in {extract_time:.1f}s ({total_images/extract_time:.0f} img/s)')
        
        # Stage 3: Tokenize
        print('\nStage 3: Tokenizing...')
        token_start = time.time()
        
        success = tokenize_directory(args.tokenizer, args.preset, img_dir, args.output)
        
        token_time = time.time() - token_start
        
        if success:
            file_size = os.path.getsize(args.output) / (1024 * 1024)
            print(f'  Tokenized in {token_time:.1f}s ({total_images/token_time:.0f} img/s)')
        else:
            print('  Tokenization failed!')
            return 1
    
    total_time = time.time() - total_start
    
    print(f'\n{"="*50}')
    print(f'COMPLETE')
    print(f'{"="*50}')
    print(f'Total images: {total_images}')
    print(f'Total time: {total_time:.1f}s')
    print(f'Effective rate: {total_images/total_time:.0f} img/s')
    print(f'Output: {args.output} ({file_size:.1f} MB)')
    print(f'\nBreakdown:')
    print(f'  Download: {download_time:.1f}s ({100*download_time/total_time:.0f}%)')
    print(f'  Extract:  {extract_time:.1f}s ({100*extract_time/total_time:.0f}%)')
    print(f'  Tokenize: {token_time:.1f}s ({100*token_time/total_time:.0f}%)')

if __name__ == '__main__':
    main()
