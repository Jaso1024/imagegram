#!/usr/bin/env python3
"""
Fast wavelet tokenization of ImageNet using multiprocessing.
"""

import numpy as np
from PIL import Image
import pywt
from pathlib import Path
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import io

# Global tokenizer for worker processes
_tokenizer = None

class WaveletTokenizer:
    """Wavelet tokenizer with fixed quantization."""
    
    PRESETS = {
        'tiny':   (256, 'haar', 6, 0, 10),   # 48 tokens
        'small':  (256, 'haar', 6, 1, 10),   # 192 tokens  
        'medium': (256, 'haar', 5, 1, 10),   # 768 tokens
        'large':  (256, 'haar', 4, 2, 10),   # 12288 tokens
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
    
    def _quantize_approx(self, coeffs):
        lo, hi = self.approx_range
        normalized = (coeffs - lo) / (hi - lo + 1e-8)
        tokens = np.clip(normalized * (self.vocab_size - 1), 0, self.vocab_size - 1)
        return tokens.astype(np.uint16)
    
    def _quantize_detail(self, coeffs):
        lo, hi = self.detail_range
        coeffs = np.clip(coeffs, lo, hi)
        normalized = (coeffs - lo) / (hi - lo + 1e-8)
        tokens = np.clip(normalized * (self.vocab_size - 1), 0, self.vocab_size - 1)
        return tokens.astype(np.uint16)
    
    def tokenize(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] != (self.image_size, self.image_size):
            img = Image.fromarray(image)
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
            image = np.array(img)
        
        all_tokens = []
        
        for c in range(3):
            channel = image[:, :, c].astype(np.float32)
            coeffs = pywt.wavedec2(channel, self.wavelet, level=self.levels)
            
            approx = coeffs[0]
            approx_tokens = self._quantize_approx(approx.flatten())
            all_tokens.extend(approx_tokens)
            
            for level_idx in range(1, self.detail_levels + 1):
                cH, cV, cD = coeffs[level_idx]
                h, v, d = cH.flatten(), cV.flatten(), cD.flatten()
                detail = np.stack([h, v, d], axis=1).flatten()
                detail_tokens = self._quantize_detail(detail)
                all_tokens.extend(detail_tokens)
        
        return np.array(all_tokens, dtype=np.uint16)
    
    def tokenize_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Tokenize from raw bytes."""
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        return self.tokenize(np.array(img))


def init_worker(preset):
    """Initialize worker with tokenizer."""
    global _tokenizer
    _tokenizer = WaveletTokenizer(preset)


def tokenize_file(path):
    """Tokenize a single file."""
    global _tokenizer
    try:
        img = Image.open(path).convert('RGB')
        img = img.resize((_tokenizer.image_size, _tokenizer.image_size), Image.LANCZOS)
        tokens = _tokenizer.tokenize(np.array(img))
        return tokens
    except Exception as e:
        return None


def tokenize_imagenet_local(input_dir: str, output_file: str, preset: str, num_workers: int = None):
    """Tokenize ImageNet from local directory."""
    tokenizer = WaveletTokenizer(preset)
    
    print(f"Tokenizer: {preset}")
    print(f"  Tokens per image: {tokenizer.tokens_per_image}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    
    # Find all images
    input_path = Path(input_dir)
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG']:
        image_paths.extend(input_path.rglob(ext))
    
    print(f"Found {len(image_paths)} images")
    
    if num_workers is None:
        num_workers = os.cpu_count()
    
    print(f"Using {num_workers} workers")
    
    # Process in parallel
    all_tokens = []
    start = time.time()
    
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(preset,)) as executor:
        futures = {executor.submit(tokenize_file, p): p for p in image_paths}
        
        done = 0
        for future in as_completed(futures):
            tokens = future.result()
            if tokens is not None:
                all_tokens.append(tokens)
            
            done += 1
            if done % 1000 == 0:
                elapsed = time.time() - start
                rate = done / elapsed
                print(f"\r{done}/{len(image_paths)} ({rate:.0f} img/s)", end='', flush=True)
    
    elapsed = time.time() - start
    print(f"\n\nProcessed {len(all_tokens)} images in {elapsed:.1f}s ({len(all_tokens)/elapsed:.0f} img/s)")
    
    # Save
    print(f"Saving to {output_file}...")
    with open(output_file, 'wb') as f:
        separator = np.array([65535], dtype=np.uint16)
        for tokens in all_tokens:
            f.write(tokens.tobytes())
            f.write(separator.tobytes())
    
    # Save config
    config = {
        'preset': preset,
        'tokens_per_image': tokenizer.tokens_per_image,
        'vocab_size': tokenizer.vocab_size,
        'separator_token': 65535,
        'num_images': len(all_tokens),
        'tokenizer': 'wavelet_v2',
        'approx_size': tokenizer.approx_size,
        'levels': tokenizer.levels,
        'detail_levels': tokenizer.detail_levels,
    }
    
    config_path = output_file.replace('.bin', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved config to {config_path}")
    
    file_size = os.path.getsize(output_file)
    print(f"Output file size: {file_size / 1e9:.2f} GB")


def tokenize_hf_streaming(output_file: str, preset: str, max_images: int = None, num_workers: int = 32):
    """Tokenize ImageNet from HuggingFace streaming."""
    from datasets import load_dataset
    
    tokenizer = WaveletTokenizer(preset)
    
    print(f"Tokenizer: {preset}")
    print(f"  Tokens per image: {tokenizer.tokens_per_image}")
    
    print("Loading ImageNet dataset (streaming)...")
    ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
    
    # Process in batches
    batch_size = 1000
    all_tokens = []
    start = time.time()
    count = 0
    
    batch = []
    for sample in ds:
        img = sample['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((tokenizer.image_size, tokenizer.image_size), Image.LANCZOS)
        batch.append(np.array(img))
        
        if len(batch) >= batch_size:
            # Process batch
            for image in batch:
                tokens = tokenizer.tokenize(image)
                all_tokens.append(tokens)
            
            count += len(batch)
            elapsed = time.time() - start
            print(f"\r{count} images ({count/elapsed:.0f} img/s)", end='', flush=True)
            batch = []
            
            if max_images and count >= max_images:
                break
    
    # Process remaining
    for image in batch:
        tokens = tokenizer.tokenize(image)
        all_tokens.append(tokens)
    
    elapsed = time.time() - start
    print(f"\n\nProcessed {len(all_tokens)} images in {elapsed:.1f}s")
    
    # Save
    print(f"Saving to {output_file}...")
    with open(output_file, 'wb') as f:
        separator = np.array([65535], dtype=np.uint16)
        for tokens in all_tokens:
            f.write(tokens.tobytes())
            f.write(separator.tobytes())
    
    config = {
        'preset': preset,
        'tokens_per_image': tokenizer.tokens_per_image,
        'vocab_size': tokenizer.vocab_size,
        'separator_token': 65535,
        'num_images': len(all_tokens),
    }
    
    config_path = output_file.replace('.bin', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Input directory')
    parser.add_argument('--output', required=True, help='Output file')
    parser.add_argument('--preset', default='medium')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--hf', action='store_true', help='Use HuggingFace streaming')
    parser.add_argument('--max-images', type=int, help='Max images to process')
    args = parser.parse_args()
    
    if args.hf:
        tokenize_hf_streaming(args.output, args.preset, args.max_images, args.workers or 32)
    elif args.input:
        tokenize_imagenet_local(args.input, args.output, args.preset, args.workers)
    else:
        print("Specify --input DIR or --hf")
