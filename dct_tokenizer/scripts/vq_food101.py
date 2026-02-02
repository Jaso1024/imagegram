#!/usr/bin/env python3
"""
Train VQ codebook on Food101 dataset (public, 101k images).
"""

import numpy as np
from PIL import Image
from datasets import load_dataset
import json
import time
import argparse

def train_codebook(output_path: str, preset: str = 'medium', max_images: int = 50000):
    """Train VQ codebook on Food101."""
    
    PRESETS = {
        'tiny':   (256, 32, 256),   # 64 tokens
        'small':  (256, 16, 512),   # 256 tokens
        'medium': (256, 8, 1024),   # 1024 tokens
        'large':  (256, 4, 2048),   # 4096 tokens
    }
    
    image_size, block_size, vocab_size = PRESETS[preset]
    block_dim = block_size * block_size * 3
    
    print(f"Training VQ: {preset}")
    print(f"  Block: {block_size}x{block_size}, vocab: {vocab_size}")
    
    print("\nLoading Food101...")
    ds = load_dataset('food101', split='train', streaming=True)
    
    # Collect blocks
    all_blocks = []
    count = 0
    start = time.time()
    
    for sample in ds:
        img = sample['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((image_size, image_size), Image.LANCZOS)
        img = np.array(img)
        
        for y in range(0, image_size, block_size):
            for x in range(0, image_size, block_size):
                block = img[y:y+block_size, x:x+block_size].flatten()
                all_blocks.append(block)
        
        count += 1
        if count % 1000 == 0:
            elapsed = time.time() - start
            print(f"\r  {count} images ({count/elapsed:.0f} img/s)", end='', flush=True)
        
        if count >= max_images:
            break
    
    print(f"\n  Collected {len(all_blocks)} blocks from {count} images")
    
    all_blocks = np.array(all_blocks, dtype=np.float32)
    
    # Subsample
    max_blocks = 500000
    if len(all_blocks) > max_blocks:
        idx = np.random.choice(len(all_blocks), max_blocks, replace=False)
        all_blocks = all_blocks[idx]
        print(f"  Subsampled to {len(all_blocks)} blocks")
    
    # K-means
    print(f"\nTraining k-means ({vocab_size} clusters)...")
    
    idx = np.random.choice(len(all_blocks), vocab_size, replace=False)
    codebook = all_blocks[idx].copy()
    
    for iteration in range(20):
        # Assign
        blocks_sq = np.sum(all_blocks ** 2, axis=1, keepdims=True)
        codebook_sq = np.sum(codebook ** 2, axis=1)
        cross = all_blocks @ codebook.T
        dists = blocks_sq + codebook_sq - 2 * cross
        assignments = np.argmin(dists, axis=1)
        
        # Update
        new_codebook = np.zeros_like(codebook)
        counts = np.zeros(vocab_size)
        
        for i, code in enumerate(assignments):
            new_codebook[code] += all_blocks[i]
            counts[code] += 1
        
        empty = counts == 0
        counts = np.maximum(counts, 1)
        new_codebook /= counts[:, np.newaxis]
        
        if empty.any():
            new_codebook[empty] = all_blocks[np.random.choice(len(all_blocks), empty.sum())]
        
        codebook = new_codebook
        
        selected = codebook[assignments]
        distortion = np.mean(np.sqrt(np.sum((all_blocks - selected) ** 2, axis=1)))
        print(f"  Iter {iteration+1}: RMSE = {distortion:.1f}")
    
    # Save
    np.savez(output_path, 
             codebook=codebook,
             preset=preset,
             image_size=image_size,
             block_size=block_size,
             vocab_size=vocab_size)
    
    print(f"\nSaved to {output_path}")
    return codebook


def tokenize_food101(codebook_path: str, output_path: str, max_images: int = None):
    """Tokenize Food101."""
    
    data = np.load(codebook_path)
    codebook = data['codebook'].astype(np.float32)
    image_size = int(data['image_size'])
    block_size = int(data['block_size'])
    vocab_size = len(codebook)
    tokens_per_image = (image_size // block_size) ** 2
    
    print(f"Tokenizing Food101")
    print(f"  Tokens per image: {tokens_per_image}, vocab: {vocab_size}")
    
    ds = load_dataset('food101', split='train', streaming=True)
    
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
            
            blocks = []
            for y in range(0, image_size, block_size):
                for x in range(0, image_size, block_size):
                    blocks.append(img[y:y+block_size, x:x+block_size].flatten())
            
            blocks = np.array(blocks, dtype=np.float32)
            
            # Quantize
            blocks_sq = np.sum(blocks ** 2, axis=1, keepdims=True)
            codebook_sq = np.sum(codebook ** 2, axis=1)
            dists = blocks_sq + codebook_sq - 2 * (blocks @ codebook.T)
            tokens = np.argmin(dists, axis=1).astype(np.uint16)
            
            f.write(tokens.tobytes())
            f.write(separator.tobytes())
            
            count += 1
            if count % 1000 == 0:
                elapsed = time.time() - start
                print(f"\r{count} ({count/elapsed:.0f} img/s)", end='', flush=True)
            
            if max_images and count >= max_images:
                break
    
    elapsed = time.time() - start
    print(f"\n\nTokenized {count} images in {elapsed:.1f}s")
    
    # Config
    config = {
        'tokenizer': 'simple_vq',
        'preset': str(data['preset']),
        'tokens_per_image': tokens_per_image,
        'vocab_size': vocab_size,
        'separator_token': 65535,
        'num_images': count,
        'block_size': block_size,
        'image_size': image_size,
        'dataset': 'food101',
    }
    
    with open(output_path.replace('.bin', '_config.json'), 'w') as f:
        json.dump(config, f, indent=2)


def test_roundtrip(codebook_path: str):
    """Test reconstruction quality."""
    
    data = np.load(codebook_path)
    codebook = data['codebook'].astype(np.float32)
    image_size = int(data['image_size'])
    block_size = int(data['block_size'])
    
    print(f"Testing codebook: {codebook.shape}")
    
    ds = load_dataset('food101', split='train', streaming=True)
    
    psnr_total = 0
    n_test = 100
    
    for i, sample in enumerate(ds):
        if i >= n_test:
            break
        
        img = sample['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((image_size, image_size), Image.LANCZOS)
        original = np.array(img)
        
        # Tokenize
        blocks = []
        for y in range(0, image_size, block_size):
            for x in range(0, image_size, block_size):
                blocks.append(original[y:y+block_size, x:x+block_size].flatten())
        blocks = np.array(blocks, dtype=np.float32)
        
        blocks_sq = np.sum(blocks ** 2, axis=1, keepdims=True)
        codebook_sq = np.sum(codebook ** 2, axis=1)
        dists = blocks_sq + codebook_sq - 2 * (blocks @ codebook.T)
        tokens = np.argmin(dists, axis=1)
        
        # Detokenize
        recon_blocks = codebook[tokens]
        recon = np.zeros((image_size, image_size, 3), dtype=np.float32)
        
        idx = 0
        for y in range(0, image_size, block_size):
            for x in range(0, image_size, block_size):
                recon[y:y+block_size, x:x+block_size] = recon_blocks[idx].reshape(block_size, block_size, 3)
                idx += 1
        
        recon = np.clip(recon, 0, 255)
        
        mse = np.mean((original.astype(float) - recon) ** 2)
        psnr = 10 * np.log10(255**2 / (mse + 1e-10))
        psnr_total += psnr
    
    print(f"Average PSNR over {n_test} images: {psnr_total/n_test:.2f} dB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--tokenize', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--codebook', default='/root/vq_food101_codebook.npz')
    parser.add_argument('--output', default='/root/vq_food101_tokens.bin')
    parser.add_argument('--preset', default='medium')
    parser.add_argument('--max-images', type=int, default=75750)  # All of food101 train
    args = parser.parse_args()
    
    if args.train:
        train_codebook(args.codebook, args.preset, min(args.max_images, 50000))
    
    if args.test:
        test_roundtrip(args.codebook)
    
    if args.tokenize:
        tokenize_food101(args.codebook, args.output, args.max_images)
