#!/usr/bin/env python3
"""
Fast webdataset tokenization pipeline.
Downloads shards in parallel, tokenizes on the fly.
"""

import os
import io
import tarfile
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import queue
import requests

# Pyramid tokenizer (inline to avoid import issues)
class PyramidTokenizer:
    PRESETS = {
        'tiny':   (256, [4, 8], 8),
        'small':  (256, [8, 16], 8),
        'medium': (256, [8, 16, 32], 8),
        'large':  (256, [16, 32, 64], 8),
    }
    
    def __init__(self, preset='small'):
        cfg = self.PRESETS[preset]
        self.output_size = cfg[0]
        self.encoded_levels = cfg[1]
        self.bits = cfg[2]
        self.vocab_size = 1 << self.bits
        self.tokens_per_level = [s * s * 3 for s in self.encoded_levels]
        self.tokens_per_image = sum(self.tokens_per_level)
    
    def tokenize(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] != (self.output_size, self.output_size):
            img = Image.fromarray(image)
            img = img.resize((self.output_size, self.output_size), Image.LANCZOS)
            image = np.array(img)
        
        image = image.astype(np.float32)
        tokens = []
        
        for size in self.encoded_levels:
            img_pil = Image.fromarray(image.astype(np.uint8))
            img_pil = img_pil.resize((size, size), Image.BILINEAR)
            level = np.array(img_pil).astype(np.float32)
            level_tokens = np.clip(level, 0, 255).astype(np.uint8).flatten()
            tokens.extend(level_tokens)
        
        return np.array(tokens, dtype=np.uint16)


def download_shard(url: str, timeout: int = 300) -> bytes:
    """Download a shard tar file."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def process_shard(tar_bytes: bytes, tokenizer: PyramidTokenizer) -> list:
    """Extract and tokenize all images from a shard."""
    tokens_list = []
    
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.jpg') or member.name.endswith('.png'):
                try:
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    
                    img = Image.open(io.BytesIO(f.read()))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    tokens = tokenizer.tokenize(np.array(img))
                    tokens_list.append(tokens)
                except Exception as e:
                    pass  # Skip bad images
    
    return tokens_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='/root/cc12m_tokens.bin')
    parser.add_argument('--preset', default='small')
    parser.add_argument('--shards', type=int, default=100, help='Number of shards to process')
    parser.add_argument('--start-shard', type=int, default=0)
    parser.add_argument('--download-workers', type=int, default=8)
    parser.add_argument('--process-workers', type=int, default=32)
    args = parser.parse_args()
    
    tokenizer = PyramidTokenizer(args.preset)
    print(f"Tokenizer: {args.preset}, {tokenizer.tokens_per_image} tokens/image")
    
    base_url = "https://huggingface.co/datasets/laion/conceptual-captions-12m-webdataset/resolve/main/data"
    
    shard_urls = [
        f"{base_url}/{i:05d}.tar" 
        for i in range(args.start_shard, args.start_shard + args.shards)
    ]
    
    print(f"Processing {len(shard_urls)} shards...")
    
    separator = np.array([65535], dtype=np.uint16)
    total_images = 0
    start_time = time.time()
    
    # Queue for downloaded shards waiting to be processed
    shard_queue = queue.Queue(maxsize=args.download_workers * 2)
    results_queue = queue.Queue()
    done_downloading = threading.Event()
    
    def downloader():
        """Download shards and put in queue."""
        with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
            futures = {executor.submit(download_shard, url): url for url in shard_urls}
            
            for future in as_completed(futures):
                url = futures[future]
                try:
                    data = future.result()
                    shard_queue.put(data)
                except Exception as e:
                    print(f"\nFailed to download {url}: {e}")
        
        done_downloading.set()
    
    def processor():
        """Process shards from queue."""
        while True:
            try:
                data = shard_queue.get(timeout=1)
                tokens_list = process_shard(data, tokenizer)
                results_queue.put(tokens_list)
            except queue.Empty:
                if done_downloading.is_set() and shard_queue.empty():
                    break
    
    # Start downloader thread
    download_thread = threading.Thread(target=downloader)
    download_thread.start()
    
    # Start processor threads
    process_threads = []
    for _ in range(args.process_workers):
        t = threading.Thread(target=processor)
        t.start()
        process_threads.append(t)
    
    # Write results as they come in
    shards_done = 0
    with open(args.output, 'wb') as f:
        while shards_done < len(shard_urls):
            try:
                tokens_list = results_queue.get(timeout=5)
                
                for tokens in tokens_list:
                    f.write(tokens.tobytes())
                    f.write(separator.tobytes())
                    total_images += 1
                
                shards_done += 1
                elapsed = time.time() - start_time
                rate = total_images / elapsed
                print(f"\rShards: {shards_done}/{len(shard_urls)}, Images: {total_images:,}, Rate: {rate:.0f} img/s", end='', flush=True)
                
            except queue.Empty:
                continue
    
    # Wait for threads
    download_thread.join()
    for t in process_threads:
        t.join()
    
    elapsed = time.time() - start_time
    print(f"\n\nDone! {total_images:,} images in {elapsed:.1f}s ({total_images/elapsed:.0f} img/s)")
    
    # Save config
    config = {
        'tokenizer': 'pyramid',
        'preset': args.preset,
        'tokens_per_image': tokenizer.tokens_per_image,
        'vocab_size': tokenizer.vocab_size,
        'separator_token': 65535,
        'num_images': total_images,
        'dataset': 'cc12m',
        'shards': args.shards,
    }
    
    config_path = args.output.replace('.bin', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Config: {config_path}")


if __name__ == '__main__':
    main()
