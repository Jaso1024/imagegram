#!/usr/bin/env python3
"""
Fast pipeline v2: Overlap download and extraction using producer-consumer pattern.
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
from threading import Thread, Event
import struct

def download_worker(shard_ids, base_url, headers, num_total_shards, output_queue, done_event):
    """Download parquets and put them in queue."""
    for shard_id in shard_ids:
        url = f'{base_url}/train-{shard_id:05d}-of-{num_total_shards:05d}.parquet'
        try:
            r = requests.get(url, headers=headers, timeout=300)
            r.raise_for_status()
            output_queue.put((shard_id, r.content))
        except Exception as e:
            print(f'Download error shard {shard_id}: {e}', file=sys.stderr)
    done_event.set()

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

def extract_worker(input_queue, output_dir, done_event, stats):
    """Extract images from parquets in queue."""
    executor = ThreadPoolExecutor(max_workers=64)
    
    while not (done_event.is_set() and input_queue.empty()):
        try:
            shard_id, parquet_bytes = input_queue.get(timeout=1)
        except:
            continue
        
        table = pq.read_table(io.BytesIO(parquet_bytes))
        start_idx = shard_id * 50000  # Approximate, just needs to be unique
        
        args_list = []
        for i in range(len(table)):
            img_data = table['image'][i].as_py()
            output_path = output_dir / f'{start_idx + i:08d}.jpg'
            args_list.append((img_data, output_path))
        
        results = list(executor.map(extract_single_image, args_list))
        count = sum(results)
        
        stats['images'] += count
        stats['shards'] += 1
        print(f'  Shard {shard_id}: {count} images (total: {stats["images"]})')
    
    executor.shutdown()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--preset', default='small')
    parser.add_argument('--num-shards', type=int, default=8)
    parser.add_argument('--parallel-downloads', type=int, default=4)
    parser.add_argument('--tokenizer', default='/root/dct_tokenizer/build/tokenize_compact')
    parser.add_argument('--dataset', default='benjamin-paine/imagenet-1k-256x256')
    args = parser.parse_args()
    
    hf_token = os.environ.get('HF_TOKEN', '')
    headers = {'Authorization': f'Bearer {hf_token}'} if hf_token else {}
    
    num_total_shards = 40
    base_url = f'https://huggingface.co/datasets/{args.dataset}/resolve/main/data'
    
    print(f'Dataset: {args.dataset}')
    print(f'Downloading {args.num_shards} shards')
    print(f'Preset: {args.preset}')
    print()
    
    total_start = time.time()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        img_dir = tmpdir / 'images'
        img_dir.mkdir()
        
        # Setup queue and events
        parquet_queue = Queue(maxsize=4)
        download_done = Event()
        stats = {'images': 0, 'shards': 0}
        
        # Split shards among download workers
        shard_lists = [[] for _ in range(args.parallel_downloads)]
        for i, shard_id in enumerate(range(args.num_shards)):
            shard_lists[i % args.parallel_downloads].append(shard_id)
        
        print('Starting download + extraction pipeline...')
        pipeline_start = time.time()
        
        # Start download threads
        download_threads = []
        download_events = []
        for shards in shard_lists:
            if shards:
                event = Event()
                download_events.append(event)
                t = Thread(target=download_worker, args=(shards, base_url, headers, num_total_shards, parquet_queue, event))
                t.start()
                download_threads.append(t)
        
        # Start extract thread
        def all_downloads_done():
            return all(e.is_set() for e in download_events)
        
        extract_thread = Thread(target=extract_worker, args=(parquet_queue, img_dir, Event(), stats))
        extract_thread.start()
        
        # Wait for downloads
        for t in download_threads:
            t.join()
        
        # Signal extraction to finish
        download_done.set()
        
        # Wait for extraction (with timeout)
        while stats['shards'] < args.num_shards:
            time.sleep(0.5)
        
        pipeline_time = time.time() - pipeline_start
        print(f'\nPipeline complete: {stats["images"]} images in {pipeline_time:.1f}s ({stats["images"]/pipeline_time:.0f} img/s)')
        
        # Tokenize
        print('\nTokenizing...')
        token_start = time.time()
        
        cmd = [args.tokenizer, args.preset, str(img_dir), args.output]
        result = subprocess.run(cmd, capture_output=True)
        
        token_time = time.time() - token_start
        
        if result.returncode == 0:
            file_size = os.path.getsize(args.output) / (1024 * 1024)
            print(f'Tokenized in {token_time:.1f}s ({stats["images"]/token_time:.0f} img/s)')
        else:
            print(f'Tokenization failed: {result.stderr.decode()}')
            return 1
    
    total_time = time.time() - total_start
    
    print(f'\n{"="*50}')
    print(f'COMPLETE')
    print(f'{"="*50}')
    print(f'Total images: {stats["images"]}')
    print(f'Total time: {total_time:.1f}s')
    print(f'Effective rate: {stats["images"]/total_time:.0f} img/s')
    print(f'Output: {args.output} ({file_size:.1f} MB)')

if __name__ == '__main__':
    main()
