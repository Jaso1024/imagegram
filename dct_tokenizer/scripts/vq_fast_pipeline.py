#!/usr/bin/env python3
"""
Fast VQ tokenization pipeline for ImageNet.
Uses multiprocessing and trained codebook.
"""

import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import queue
import threading

# Worker globals
_vq = None
_codebook = None
_image_size = 256
_block_size = 8
_vocab_size = 1024

def init_worker(codebook_path):
    """Initialize worker with codebook."""
    global _vq, _codebook, _image_size, _block_size, _vocab_size
    
    data = np.load(codebook_path)
    _codebook = data['codebook'].astype(np.float32)
    _vocab_size = len(_codebook)
    _block_size = int(np.cbrt(_codebook.shape[1] / 3))  # Approximate
    
    # Determine block size from codebook dimension
    block_dim = _codebook.shape[1]
    for bs in [4, 8, 16, 32]:
        if bs * bs * 3 == block_dim:
            _block_size = bs
            break
    
    _image_size = 256


def tokenize_image(image_array):
    """Tokenize a single image."""
    global _codebook, _image_size, _block_size
    
    img = image_array
    if img.shape[:2] != (_image_size, _image_size):
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((_image_size, _image_size), Image.LANCZOS)
        img = np.array(img_pil)
    
    # Extract blocks
    blocks_per_side = _image_size // _block_size
    blocks = []
    
    for y in range(0, _image_size, _block_size):
        for x in range(0, _image_size, _block_size):
            block = img[y:y+_block_size, x:x+_block_size]
            blocks.append(block.flatten())
    
    blocks = np.array(blocks, dtype=np.float32)
    
    # Fast distance computation
    blocks_sq = np.sum(blocks ** 2, axis=1, keepdims=True)
    codebook_sq = np.sum(_codebook ** 2, axis=1)
    cross = blocks @ _codebook.T
    dists = blocks_sq + codebook_sq - 2 * cross
    
    tokens = np.argmin(dists, axis=1).astype(np.uint16)
    return tokens


def train_codebook_on_imagenet(output_path: str, preset: str = 'medium', max_images: int = 50000):
    """Train VQ codebook on ImageNet images via streaming."""
    from datasets import load_dataset
    
    PRESETS = {
        'tiny':   (256, 32, 256),
        'small':  (256, 16, 512),
        'medium': (256, 8, 1024),
        'large':  (256, 4, 2048),
    }
    
    image_size, block_size, vocab_size = PRESETS[preset]
    block_dim = block_size * block_size * 3
    blocks_per_image = (image_size // block_size) ** 2
    
    print(f"Training VQ codebook: {preset}")
    print(f"  Block size: {block_size}, vocab: {vocab_size}")
    print(f"  Blocks per image: {blocks_per_image}")
    
    # Collect blocks from streaming dataset
    print(f"\nStreaming ImageNet (up to {max_images} images)...")
    
    hf_token = os.environ.get('HF_TOKEN')
    ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True, token=hf_token)
    
    all_blocks = []
    count = 0
    
    for sample in ds:
        img = sample['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((image_size, image_size), Image.LANCZOS)
        img = np.array(img)
        
        # Extract blocks
        for y in range(0, image_size, block_size):
            for x in range(0, image_size, block_size):
                block = img[y:y+block_size, x:x+block_size].flatten()
                all_blocks.append(block)
        
        count += 1
        if count % 1000 == 0:
            print(f"\r  Processed {count} images...", end='', flush=True)
        
        if count >= max_images:
            break
    
    print(f"\n  Collected {len(all_blocks)} blocks from {count} images")
    
    all_blocks = np.array(all_blocks, dtype=np.float32)
    
    # Subsample if too many
    max_blocks = 500000
    if len(all_blocks) > max_blocks:
        idx = np.random.choice(len(all_blocks), max_blocks, replace=False)
        all_blocks = all_blocks[idx]
        print(f"  Subsampled to {len(all_blocks)} blocks")
    
    # K-means training
    print(f"\nTraining k-means with {vocab_size} clusters...")
    
    # Initialize with random blocks
    idx = np.random.choice(len(all_blocks), vocab_size, replace=False)
    codebook = all_blocks[idx].copy()
    
    n_iter = 20
    for iteration in range(n_iter):
        # Assign blocks to codes
        blocks_sq = np.sum(all_blocks ** 2, axis=1, keepdims=True)
        codebook_sq = np.sum(codebook ** 2, axis=1)
        cross = all_blocks @ codebook.T
        dists = blocks_sq + codebook_sq - 2 * cross
        assignments = np.argmin(dists, axis=1)
        
        # Update codebook
        new_codebook = np.zeros_like(codebook)
        counts = np.zeros(vocab_size)
        
        for i, code in enumerate(assignments):
            new_codebook[code] += all_blocks[i]
            counts[code] += 1
        
        # Avoid division by zero
        empty = counts == 0
        counts = np.maximum(counts, 1)
        new_codebook /= counts[:, np.newaxis]
        
        # Handle empty codes
        if empty.any():
            n_empty = empty.sum()
            new_codebook[empty] = all_blocks[np.random.choice(len(all_blocks), n_empty)]
        
        codebook = new_codebook
        
        # Compute distortion
        selected = codebook[assignments]
        distortion = np.mean(np.sum((all_blocks - selected) ** 2, axis=1))
        print(f"  Iter {iteration+1}/{n_iter}: distortion = {distortion:.2f}")
    
    # Save codebook
    np.savez(output_path, 
             codebook=codebook,
             preset=preset,
             image_size=image_size,
             block_size=block_size,
             vocab_size=vocab_size)
    
    print(f"\nSaved codebook to {output_path}")
    print(f"  Shape: {codebook.shape}")
    
    return codebook


def tokenize_imagenet_streaming(codebook_path: str, output_path: str, max_images: int = None):
    """Tokenize ImageNet using streaming."""
    from datasets import load_dataset
    
    # Load codebook
    data = np.load(codebook_path)
    codebook = data['codebook'].astype(np.float32)
    image_size = int(data['image_size'])
    block_size = int(data['block_size'])
    vocab_size = len(codebook)
    
    blocks_per_image = (image_size // block_size) ** 2
    
    print(f"Tokenizing with codebook: {codebook_path}")
    print(f"  Block size: {block_size}, vocab: {vocab_size}")
    print(f"  Tokens per image: {blocks_per_image}")
    
    hf_token = os.environ.get('HF_TOKEN')
    ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True, token=hf_token)
    
    separator = np.array([65535], dtype=np.uint16)
    
    start = time.time()
    count = 0
    
    with open(output_path, 'wb') as f:
        for sample in ds:
            img = sample['image']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((image_size, image_size), Image.LANCZOS)
            img = np.array(img)
            
            # Extract blocks
            blocks = []
            for y in range(0, image_size, block_size):
                for x in range(0, image_size, block_size):
                    block = img[y:y+block_size, x:x+block_size].flatten()
                    blocks.append(block)
            
            blocks = np.array(blocks, dtype=np.float32)
            
            # Quantize
            blocks_sq = np.sum(blocks ** 2, axis=1, keepdims=True)
            codebook_sq = np.sum(codebook ** 2, axis=1)
            cross = blocks @ codebook.T
            dists = blocks_sq + codebook_sq - 2 * cross
            tokens = np.argmin(dists, axis=1).astype(np.uint16)
            
            f.write(tokens.tobytes())
            f.write(separator.tobytes())
            
            count += 1
            if count % 1000 == 0:
                elapsed = time.time() - start
                rate = count / elapsed
                print(f"\r{count:,} images ({rate:.0f} img/s)", end='', flush=True)
            
            if max_images and count >= max_images:
                break
    
    elapsed = time.time() - start
    print(f"\n\nTokenized {count:,} images in {elapsed:.1f}s ({count/elapsed:.0f} img/s)")
    
    # Save config
    config = {
        'tokenizer': 'simple_vq',
        'preset': str(data.get('preset', 'medium')),
        'tokens_per_image': blocks_per_image,
        'vocab_size': vocab_size,
        'separator_token': 65535,
        'num_images': count,
        'block_size': block_size,
        'image_size': image_size,
    }
    
    config_path = output_path.replace('.bin', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved config to {config_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train codebook')
    parser.add_argument('--tokenize', action='store_true', help='Tokenize dataset')
    parser.add_argument('--codebook', default='/root/vq_imagenet_codebook.npz')
    parser.add_argument('--output', default='/root/vq_imagenet_tokens.bin')
    parser.add_argument('--preset', default='medium')
    parser.add_argument('--max-images', type=int)
    args = parser.parse_args()
    
    if args.train:
        train_codebook_on_imagenet(args.codebook, args.preset, args.max_images or 50000)
    
    if args.tokenize:
        tokenize_imagenet_streaming(args.codebook, args.output, args.max_images)


if __name__ == '__main__':
    main()
