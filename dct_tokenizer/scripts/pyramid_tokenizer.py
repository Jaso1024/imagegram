#!/usr/bin/env python3
"""
Gaussian Pyramid Tokenizer - NO TRAINING.

Strategy: Encode only coarse levels, upsample to full resolution.
This gives coarse-to-fine token ordering with spatial coherence.

Token order:
1. Base level (e.g., 8x8 RGB = 192 tokens) - global colors
2. Residual level 1 (e.g., 16x16 RGB = 768 tokens) - medium detail
3. Final reconstruction upsamples to 256x256

For n-grams: tokens are in raster order within each level,
and levels go coarse-to-fine.
"""

import numpy as np
from PIL import Image
import time
import argparse
from pathlib import Path
import json

class PyramidTokenizer:
    """Gaussian pyramid tokenizer - encode coarse, upsample to full res."""
    
    PRESETS = {
        # (output_size, encoded_levels, bits)
        # encoded_levels = list of sizes to actually encode
        'tiny':   (256, [4, 8], 8),       # 48 + 192 = 240 tokens
        'small':  (256, [8, 16], 8),      # 192 + 768 = 960 tokens  
        'medium': (256, [8, 16, 32], 8),  # 192 + 768 + 3072 = 4032 tokens
        'large':  (256, [16, 32, 64], 8), # 768 + 3072 + 12288 = 16128 tokens
    }
    
    def __init__(self, preset='small'):
        cfg = self.PRESETS[preset]
        self.output_size = cfg[0]
        self.encoded_levels = cfg[1]
        self.bits = cfg[2]
        self.preset = preset
        
        self.vocab_size = 1 << self.bits
        self.tokens_per_level = [s * s * 3 for s in self.encoded_levels]
        self.tokens_per_image = sum(self.tokens_per_level)
        
        print(f"PyramidTokenizer: {preset}")
        print(f"  Output size: {self.output_size}x{self.output_size}")
        print(f"  Encoded levels: {self.encoded_levels}")
        print(f"  Tokens per level: {self.tokens_per_level}")
        print(f"  Total tokens: {self.tokens_per_image}")
        print(f"  Vocab size: {self.vocab_size}")
    
    def tokenize(self, image: np.ndarray, use_residuals: bool = False) -> np.ndarray:
        """Tokenize via pyramid encoding.
        
        If use_residuals=False (default), encode actual pixel values at each level.
        This gives better n-gram statistics since values represent colors, not differences.
        """
        # Resize input to output size
        if image.shape[:2] != (self.output_size, self.output_size):
            img = Image.fromarray(image)
            img = img.resize((self.output_size, self.output_size), Image.LANCZOS)
            image = np.array(img)
        
        image = image.astype(np.float32)
        tokens = []
        
        # Create downsampled versions at each encoded level
        levels = []
        for size in self.encoded_levels:
            img_pil = Image.fromarray(image.astype(np.uint8))
            img_pil = img_pil.resize((size, size), Image.BILINEAR)
            levels.append(np.array(img_pil).astype(np.float32))
        
        if use_residuals:
            # Original residual encoding
            base = levels[0]
            base_tokens = np.clip(base, 0, 255).astype(np.uint8).flatten()
            tokens.extend(base_tokens)
            
            for i in range(1, len(levels)):
                prev_img = Image.fromarray(levels[i - 1].astype(np.uint8))
                curr_size = self.encoded_levels[i]
                upsampled = np.array(prev_img.resize((curr_size, curr_size), Image.BILINEAR)).astype(np.float32)
                residual = levels[i] - upsampled
                residual_q = np.clip(residual + 128, 0, 255).astype(np.uint8)
                tokens.extend(residual_q.flatten())
        else:
            # Direct pixel encoding at each level - better for n-grams
            for level in levels:
                level_tokens = np.clip(level, 0, 255).astype(np.uint8).flatten()
                tokens.extend(level_tokens)
        
        return np.array(tokens, dtype=np.uint16)
    
    def detokenize(self, tokens: np.ndarray, use_residuals: bool = False) -> np.ndarray:
        """Reconstruct from pyramid tokens, upsample to full res."""
        idx = 0
        
        if use_residuals:
            # Original residual decoding
            base_size = self.encoded_levels[0]
            base_count = self.tokens_per_level[0]
            base_tokens = tokens[idx:idx + base_count]
            idx += base_count
            
            current = base_tokens.reshape(base_size, base_size, 3).astype(np.float32)
            
            for i in range(1, len(self.encoded_levels)):
                curr_size = self.encoded_levels[i]
                level_count = self.tokens_per_level[i]
                
                img_pil = Image.fromarray(np.clip(current, 0, 255).astype(np.uint8))
                upsampled = np.array(img_pil.resize((curr_size, curr_size), Image.BILINEAR)).astype(np.float32)
                
                residual_tokens = tokens[idx:idx + level_count]
                idx += level_count
                residual = residual_tokens.reshape(curr_size, curr_size, 3).astype(np.float32) - 128
                
                current = upsampled + residual
        else:
            # Direct decoding - just use the finest level
            # Skip coarser levels and use only the last (finest) level
            for i, (size, count) in enumerate(zip(self.encoded_levels, self.tokens_per_level)):
                level_tokens = tokens[idx:idx + count]
                idx += count
                current = level_tokens.reshape(size, size, 3).astype(np.float32)
        
        # Final upsample to output size
        final_size = self.encoded_levels[-1]
        if final_size != self.output_size:
            img_pil = Image.fromarray(np.clip(current, 0, 255).astype(np.uint8))
            current = np.array(img_pil.resize((self.output_size, self.output_size), Image.BILINEAR))
        
        return np.clip(current, 0, 255).astype(np.uint8)
    
    def tokenize_batch(self, images: list) -> np.ndarray:
        """Tokenize batch of images."""
        return np.array([self.tokenize(img) for img in images])


def test_roundtrip(image_path: str, preset: str):
    """Test tokenizer."""
    tok = PyramidTokenizer(preset)
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize((tok.output_size, tok.output_size), Image.LANCZOS)
    image = np.array(img)
    
    print(f"\nTokenizing {image_path}...")
    start = time.time()
    tokens = tok.tokenize(image)
    elapsed = time.time() - start
    
    print(f"Tokens: {len(tokens)}, range [{tokens.min()}, {tokens.max()}]")
    print(f"Unique: {len(np.unique(tokens))}")
    print(f"Time: {elapsed*1000:.1f}ms")
    
    # Show tokens per level
    idx = 0
    for i, (size, count) in enumerate(zip(tok.encoded_levels, tok.tokens_per_level)):
        level_tokens = tokens[idx:idx + count]
        print(f"  Level {i} ({size}x{size}): {count} tokens, unique={len(np.unique(level_tokens))}, mean={level_tokens.mean():.1f}")
        idx += count
    
    print("\nDetokenizing...")
    recon = tok.detokenize(tokens)
    
    mse = np.mean((image.astype(float) - recon.astype(float)) ** 2)
    psnr = 10 * np.log10(255**2 / (mse + 1e-10))
    print(f"PSNR: {psnr:.2f} dB")
    
    out_path = Path(image_path).stem + f"_pyramid_{preset}.png"
    Image.fromarray(recon).save(out_path)
    print(f"Saved: {out_path}")
    
    return tokens


def speed_test(preset: str):
    """Speed test."""
    tok = PyramidTokenizer(preset)
    
    images = [np.random.randint(0, 256, (tok.output_size, tok.output_size, 3), dtype=np.uint8) for _ in range(100)]
    
    n = 5000
    print(f"\nSpeed test: {n} images...")
    
    start = time.time()
    for i in range(n):
        tok.tokenize(images[i % 100])
    
    elapsed = time.time() - start
    print(f"Rate: {n/elapsed:.0f} img/s")


def tokenize_dataset_streaming(preset: str, output_path: str, dataset_name: str = 'food101', max_images: int = None):
    """Tokenize a streaming dataset."""
    from datasets import load_dataset
    
    tok = PyramidTokenizer(preset)
    
    print(f"\nLoading {dataset_name}...")
    ds = load_dataset(dataset_name, split='train', streaming=True)
    
    separator = np.array([65535], dtype=np.uint16)
    start = time.time()
    count = 0
    
    with open(output_path, 'wb') as f:
        for sample in ds:
            img = sample['image']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((tok.output_size, tok.output_size), Image.LANCZOS)
            
            tokens = tok.tokenize(np.array(img))
            f.write(tokens.tobytes())
            f.write(separator.tobytes())
            
            count += 1
            if count % 1000 == 0:
                elapsed = time.time() - start
                print(f"\r{count} ({count/elapsed:.0f} img/s)", end='', flush=True)
            
            if max_images and count >= max_images:
                break
    
    elapsed = time.time() - start
    print(f"\n\nTokenized {count} images in {elapsed:.1f}s ({count/elapsed:.0f} img/s)")
    
    # Config
    config = {
        'tokenizer': 'pyramid',
        'preset': preset,
        'tokens_per_image': tok.tokens_per_image,
        'vocab_size': tok.vocab_size,
        'separator_token': 65535,
        'num_images': count,
        'encoded_levels': tok.encoded_levels,
        'output_size': tok.output_size,
        'dataset': dataset_name,
    }
    
    config_path = output_path.replace('.bin', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config: {config_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help='Test on image')
    parser.add_argument('--preset', default='small')
    parser.add_argument('--speed', action='store_true')
    parser.add_argument('--tokenize', help='Dataset to tokenize')
    parser.add_argument('--output', default='/root/pyramid_tokens.bin')
    parser.add_argument('--max-images', type=int)
    args = parser.parse_args()
    
    if args.test:
        test_roundtrip(args.test, args.preset)
    elif args.speed:
        speed_test(args.preset)
    elif args.tokenize:
        tokenize_dataset_streaming(args.preset, args.output, args.tokenize, args.max_images)
    else:
        for preset in ['tiny', 'small', 'medium', 'large']:
            print(f"\n{'='*50}")
            PyramidTokenizer(preset)
