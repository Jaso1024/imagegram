#!/usr/bin/env python3
"""
Simple Vector Quantization for images.

Approach:
1. Divide image into non-overlapping blocks (e.g., 16x16)
2. Each block → flatten → VQ → token
3. Tokens in raster order

For n-grams: adjacent tokens = adjacent blocks, so there's spatial coherence.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import json
import time

class SimpleVQ:
    """Simple VQ tokenizer using k-means style codebook."""
    
    PRESETS = {
        # (image_size, block_size, codebook_size)
        'tiny':   (256, 32, 256),   # 8x8 = 64 blocks, 256 codes
        'small':  (256, 16, 512),   # 16x16 = 256 blocks, 512 codes  
        'medium': (256, 8, 1024),   # 32x32 = 1024 blocks, 1024 codes
        'large':  (256, 4, 2048),   # 64x64 = 4096 blocks, 2048 codes
    }
    
    def __init__(self, preset='small'):
        cfg = self.PRESETS[preset]
        self.image_size = cfg[0]
        self.block_size = cfg[1]
        self.vocab_size = cfg[2]
        self.preset = preset
        
        self.blocks_per_side = self.image_size // self.block_size
        self.tokens_per_image = self.blocks_per_side ** 2
        self.block_dim = self.block_size * self.block_size * 3  # RGB
        
        # Initialize codebook randomly (will be refined during use)
        self.codebook = None
        
        print(f"SimpleVQ: {preset}")
        print(f"  Block size: {self.block_size}")
        print(f"  Blocks per side: {self.blocks_per_side}")
        print(f"  Tokens per image: {self.tokens_per_image}")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Block dimension: {self.block_dim}")
    
    def init_codebook_random(self):
        """Initialize codebook with random colors."""
        # Random RGB values
        self.codebook = np.random.randint(0, 256, (self.vocab_size, self.block_dim)).astype(np.float32)
    
    def init_codebook_grid(self):
        """Initialize codebook with a grid of uniform color blocks."""
        # Create uniform color patches
        n_per_dim = int(np.cbrt(self.vocab_size)) + 1
        
        colors = []
        for r in np.linspace(0, 255, n_per_dim):
            for g in np.linspace(0, 255, n_per_dim):
                for b in np.linspace(0, 255, n_per_dim):
                    # Uniform color block
                    block = np.zeros(self.block_dim)
                    for i in range(self.block_size * self.block_size):
                        block[i * 3] = r
                        block[i * 3 + 1] = g
                        block[i * 3 + 2] = b
                    colors.append(block)
                    if len(colors) >= self.vocab_size:
                        break
                if len(colors) >= self.vocab_size:
                    break
            if len(colors) >= self.vocab_size:
                break
        
        self.codebook = np.array(colors[:self.vocab_size], dtype=np.float32)
        print(f"  Initialized codebook with {len(self.codebook)} uniform color codes")
    
    def extract_blocks(self, image: np.ndarray) -> np.ndarray:
        """Extract blocks from image in raster order."""
        h, w = image.shape[:2]
        blocks = []
        
        for y in range(0, h, self.block_size):
            for x in range(0, w, self.block_size):
                block = image[y:y+self.block_size, x:x+self.block_size]
                blocks.append(block.flatten())
        
        return np.array(blocks, dtype=np.float32)
    
    def blocks_to_image(self, blocks: np.ndarray) -> np.ndarray:
        """Reconstruct image from blocks."""
        image = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
        
        idx = 0
        for y in range(self.blocks_per_side):
            for x in range(self.blocks_per_side):
                block = blocks[idx].reshape(self.block_size, self.block_size, 3)
                image[y*self.block_size:(y+1)*self.block_size, 
                      x*self.block_size:(x+1)*self.block_size] = block
                idx += 1
        
        return image
    
    def quantize_block(self, block: np.ndarray) -> int:
        """Find nearest codebook entry for a block."""
        # L2 distance
        dists = np.sum((self.codebook - block) ** 2, axis=1)
        return int(np.argmin(dists))
    
    def tokenize(self, image: np.ndarray) -> np.ndarray:
        """Tokenize an image."""
        if self.codebook is None:
            self.init_codebook_grid()
        
        if image.shape[:2] != (self.image_size, self.image_size):
            img_pil = Image.fromarray(image)
            img_pil = img_pil.resize((self.image_size, self.image_size), Image.LANCZOS)
            image = np.array(img_pil)
        
        blocks = self.extract_blocks(image)
        tokens = np.array([self.quantize_block(b) for b in blocks], dtype=np.uint16)
        
        return tokens
    
    def tokenize_fast(self, image: np.ndarray) -> np.ndarray:
        """Faster tokenization using batch distance computation."""
        if self.codebook is None:
            self.init_codebook_grid()
        
        if image.shape[:2] != (self.image_size, self.image_size):
            img_pil = Image.fromarray(image)
            img_pil = img_pil.resize((self.image_size, self.image_size), Image.LANCZOS)
            image = np.array(img_pil)
        
        blocks = self.extract_blocks(image)  # (n_blocks, block_dim)
        
        # Compute all distances at once
        # ||b - c||^2 = ||b||^2 + ||c||^2 - 2*b.c
        blocks_sq = np.sum(blocks ** 2, axis=1, keepdims=True)  # (n_blocks, 1)
        codebook_sq = np.sum(self.codebook ** 2, axis=1)  # (vocab,)
        cross = blocks @ self.codebook.T  # (n_blocks, vocab)
        
        dists = blocks_sq + codebook_sq - 2 * cross
        tokens = np.argmin(dists, axis=1).astype(np.uint16)
        
        return tokens
    
    def detokenize(self, tokens: np.ndarray) -> np.ndarray:
        """Detokenize tokens back to image."""
        if self.codebook is None:
            self.init_codebook_grid()
        
        blocks = self.codebook[tokens]
        image = self.blocks_to_image(blocks)
        
        return np.clip(image, 0, 255).astype(np.uint8)
    
    def train_codebook(self, images: list, n_iter: int = 10):
        """Train codebook using k-means on image blocks."""
        print(f"Training codebook on {len(images)} images...")
        
        # Collect all blocks
        all_blocks = []
        for img in images:
            if img.shape[:2] != (self.image_size, self.image_size):
                img_pil = Image.fromarray(img)
                img_pil = img_pil.resize((self.image_size, self.image_size), Image.LANCZOS)
                img = np.array(img_pil)
            
            blocks = self.extract_blocks(img)
            all_blocks.append(blocks)
        
        all_blocks = np.vstack(all_blocks)
        print(f"  Collected {len(all_blocks)} blocks")
        
        # Subsample if too many
        max_blocks = 100000
        if len(all_blocks) > max_blocks:
            idx = np.random.choice(len(all_blocks), max_blocks, replace=False)
            all_blocks = all_blocks[idx]
        
        # K-means style training
        # Initialize with random blocks
        idx = np.random.choice(len(all_blocks), self.vocab_size, replace=False)
        self.codebook = all_blocks[idx].copy()
        
        for iteration in range(n_iter):
            # Assign blocks to codes
            blocks_sq = np.sum(all_blocks ** 2, axis=1, keepdims=True)
            codebook_sq = np.sum(self.codebook ** 2, axis=1)
            cross = all_blocks @ self.codebook.T
            dists = blocks_sq + codebook_sq - 2 * cross
            assignments = np.argmin(dists, axis=1)
            
            # Update codebook
            new_codebook = np.zeros_like(self.codebook)
            counts = np.zeros(self.vocab_size)
            
            for i, code in enumerate(assignments):
                new_codebook[code] += all_blocks[i]
                counts[code] += 1
            
            # Avoid division by zero
            counts = np.maximum(counts, 1)
            new_codebook /= counts[:, np.newaxis]
            
            # Handle empty codes
            empty_codes = counts == 1  # Only had the initial value
            if empty_codes.any():
                # Replace with random blocks
                n_empty = empty_codes.sum()
                new_codebook[empty_codes] = all_blocks[np.random.choice(len(all_blocks), n_empty)]
            
            self.codebook = new_codebook
            
            # Compute distortion
            selected = self.codebook[assignments]
            distortion = np.mean(np.sum((all_blocks - selected) ** 2, axis=1))
            print(f"  Iter {iteration+1}/{n_iter}: distortion = {distortion:.2f}")
        
        return self
    
    def save(self, path: str):
        """Save codebook."""
        np.savez(path, codebook=self.codebook, preset=self.preset)
        print(f"Saved to {path}")
    
    def load(self, path: str):
        """Load codebook."""
        data = np.load(path)
        self.codebook = data['codebook']
        print(f"Loaded codebook: {self.codebook.shape}")
        return self


def test_roundtrip(image_path: str, preset: str):
    """Test tokenizer roundtrip."""
    vq = SimpleVQ(preset)
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize((vq.image_size, vq.image_size), Image.LANCZOS)
    image = np.array(img)
    
    print(f"\nTokenizing {image_path}...")
    start = time.time()
    tokens = vq.tokenize_fast(image)
    elapsed = time.time() - start
    
    print(f"Tokens: {len(tokens)}, range [{tokens.min()}, {tokens.max()}]")
    print(f"Unique tokens: {len(np.unique(tokens))}")
    print(f"Time: {elapsed*1000:.1f}ms")
    print(f"First 20: {tokens[:20]}")
    
    print("\nDetokenizing...")
    recon = vq.detokenize(tokens)
    
    mse = np.mean((image.astype(float) - recon.astype(float)) ** 2)
    psnr = 10 * np.log10(255**2 / (mse + 1e-10))
    print(f"PSNR: {psnr:.2f} dB")
    
    # Save
    out_path = Path(image_path).stem + f"_vq_{preset}.png"
    Image.fromarray(recon).save(out_path)
    print(f"Saved: {out_path}")
    
    return tokens


def speed_test(preset: str):
    """Test tokenization speed."""
    vq = SimpleVQ(preset)
    
    # Random images
    images = [np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(100)]
    
    n_images = 1000
    print(f"\nSpeed test: {n_images} images...")
    
    start = time.time()
    for i in range(n_images):
        tokens = vq.tokenize_fast(images[i % 100])
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Rate: {n_images/elapsed:.0f} img/s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help='Test on image')
    parser.add_argument('--preset', default='small')
    parser.add_argument('--speed', action='store_true')
    args = parser.parse_args()
    
    if args.test:
        test_roundtrip(args.test, args.preset)
    elif args.speed:
        speed_test(args.preset)
    else:
        # Quick demo
        print("Testing SimpleVQ...")
        vq = SimpleVQ('small')
        
        # Test gradient image
        x = np.linspace(0, 255, 256)
        test_img = np.zeros((256, 256, 3), dtype=np.uint8)
        test_img[:, :, 0] = x[np.newaxis, :]
        test_img[:, :, 1] = x[:, np.newaxis]
        test_img[:, :, 2] = 128
        
        tokens = vq.tokenize_fast(test_img)
        print(f"\nTokens: {len(tokens)}")
        print(f"First 20: {tokens[:20]}")
        
        recon = vq.detokenize(tokens)
        mse = np.mean((test_img.astype(float) - recon.astype(float)) ** 2)
        psnr = 10 * np.log10(255**2 / (mse + 1e-10))
        print(f"PSNR: {psnr:.2f} dB")
        
        # Speed test
        speed_test('small')
