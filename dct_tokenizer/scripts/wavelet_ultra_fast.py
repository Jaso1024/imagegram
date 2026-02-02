#!/usr/bin/env python3
"""
Ultra-fast wavelet tokenization using hf_transfer + multiprocessing.
Adapted from ultra_fast_pipeline.py
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import queue
import io

# Enable hf_transfer for fast downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import numpy as np
from PIL import Image
import pywt

class WaveletTokenizer:
    """Wavelet tokenizer - duplicated to avoid import issues in workers."""
    
    PRESETS = {
        'tiny':   (256, 'haar', 6, 0, 10),
        'small':  (256, 'haar', 6, 1, 10),
        'medium': (256, 'haar', 5, 1, 10),
        'large':  (256, 'haar', 4, 2, 10),
    }
    
    def __init__(self, preset='medium'):
        cfg = self.PRESETS[preset]
        self.image_size = cfg[0]
        self.wavelet = cfg[1]
        self.levels = cfg[2]
        self.detail_levels = cfg[3]
        self.quant_bits = cfg[4]
        self.vocab_size = 1 << self.quant_bits
        self.preset = preset
        
        self.approx_size = self.image_size // (2 ** self.levels)
        self.approx_count = self.approx_size ** 2
        
        self.detail_counts = []
        size = self.approx_size
        for _ in range(self.detail_levels):
            self.detail_counts.append(size * size * 3)
            size *= 2
        
        self.total_detail = sum(self.detail_counts)
        self.tokens_per_channel = self.approx_count + self.total_detail
        self.tokens_per_image = self.tokens_per_channel * 3
        
        scale_factor = 2 ** self.levels
        self.approx_range = (0, 255 * scale_factor)
        self.detail_range = (-300 * scale_factor / 4, 300 * scale_factor / 4)
    
    def tokenize(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] != (self.image_size, self.image_size):
            img = Image.fromarray(image)
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
            image = np.array(img)
        
        all_tokens = []
        
        for c in range(3):
            channel = image[:, :, c].astype(np.float32)
            coeffs = pywt.wavedec2(channel, self.wavelet, level=self.levels)
            
            # Approx
            approx = coeffs[0]
            lo, hi = self.approx_range
            normalized = (approx - lo) / (hi - lo + 1e-8)
            tokens = np.clip(normalized.flatten() * (self.vocab_size - 1), 0, self.vocab_size - 1)
            all_tokens.extend(tokens.astype(np.uint16))
            
            # Details
            for level_idx in range(1, self.detail_levels + 1):
                cH, cV, cD = coeffs[level_idx]
                h, v, d = cH.flatten(), cV.flatten(), cD.flatten()
                detail = np.stack([h, v, d], axis=1).flatten()
                
                lo, hi = self.detail_range
                detail = np.clip(detail, lo, hi)
                normalized = (detail - lo) / (hi - lo + 1e-8)
                tokens = np.clip(normalized * (self.vocab_size - 1), 0, self.vocab_size - 1)
                all_tokens.extend(tokens.astype(np.uint16))
        
        return np.array(all_tokens, dtype=np.uint16)


def download_and_tokenize(args):
    """Download and tokenize in streaming fashion."""
    output_file, preset, hf_token = args.output, args.preset, args.hf_token
    
    from huggingface_hub import HfFileSystem, hf_hub_download
    from datasets import load_dataset
    
    tokenizer = WaveletTokenizer(preset)
    print(f"Tokenizer: {preset}, {tokenizer.tokens_per_image} tokens/image")
    
    # Use datasets streaming for simplicity with good performance
    print("Loading ImageNet (streaming)...")
    ds = load_dataset(
        "ILSVRC/imagenet-1k", 
        split="train", 
        streaming=True,
        token=hf_token
    )
    
    # Process and save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    separator = np.array([65535], dtype=np.uint16)
    
    start = time.time()
    count = 0
    total_tokens = 0
    
    # Use threading for parallel tokenization while streaming
    token_queue = queue.Queue(maxsize=10000)
    write_done = threading.Event()
    
    def writer_thread():
        """Write tokens to file."""
        nonlocal total_tokens
        with open(output_file, 'wb') as f:
            while True:
                try:
                    tokens = token_queue.get(timeout=1)
                    if tokens is None:
                        break
                    f.write(tokens.tobytes())
                    f.write(separator.tobytes())
                    total_tokens += len(tokens)
                except queue.Empty:
                    if write_done.is_set():
                        break
    
    writer = threading.Thread(target=writer_thread)
    writer.start()
    
    # Process images
    batch = []
    batch_size = 100  # Process in small batches for memory efficiency
    
    for sample in ds:
        img = sample['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((tokenizer.image_size, tokenizer.image_size), Image.LANCZOS)
        batch.append(np.array(img))
        
        if len(batch) >= batch_size:
            # Tokenize batch
            for image in batch:
                tokens = tokenizer.tokenize(image)
                token_queue.put(tokens)
                count += 1
            
            batch = []
            
            if count % 1000 == 0:
                elapsed = time.time() - start
                rate = count / elapsed
                print(f"\r{count:,} images ({rate:.0f} img/s)", end='', flush=True)
            
            if args.max_images and count >= args.max_images:
                break
    
    # Process remaining
    for image in batch:
        tokens = tokenizer.tokenize(image)
        token_queue.put(tokens)
        count += 1
    
    # Signal writer to finish
    write_done.set()
    token_queue.put(None)
    writer.join()
    
    elapsed = time.time() - start
    print(f"\n\nProcessed {count:,} images in {elapsed:.1f}s ({count/elapsed:.0f} img/s)")
    print(f"Total tokens: {total_tokens:,}")
    
    # Save config
    config = {
        'preset': preset,
        'tokens_per_image': tokenizer.tokens_per_image,
        'vocab_size': tokenizer.vocab_size,
        'separator_token': 65535,
        'num_images': count,
        'total_tokens': total_tokens,
        'tokenizer': 'wavelet_v2',
    }
    
    config_path = output_file.replace('.bin', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved config to {config_path}")
    
    file_size = os.path.getsize(output_file)
    print(f"Output: {output_file} ({file_size/1e9:.2f} GB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='/root/wavelet_tokens.bin')
    parser.add_argument('--preset', default='medium', choices=['tiny', 'small', 'medium', 'large'])
    parser.add_argument('--hf-token', default=os.environ.get('HF_TOKEN'))
    parser.add_argument('--max-images', type=int)
    args = parser.parse_args()
    
    if not args.hf_token:
        print("Warning: No HF token provided. Set HF_TOKEN env var or use --hf-token")
    
    download_and_tokenize(args)


if __name__ == '__main__':
    main()
