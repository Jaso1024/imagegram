#!/usr/bin/env python3
"""
Simple color-based tokenizer - NO TRAINING.

Each block gets a token based on its average color.
Token = quantized RGB value.

With 10 bits: ~10 levels per channel = 1000 colors
With 12 bits: ~16 levels per channel = 4096 colors
"""

import numpy as np
from PIL import Image
import time
import argparse
from pathlib import Path

class ColorTokenizer:
    """Direct color quantization tokenizer."""
    
    PRESETS = {
        # (image_size, block_size, bits_per_channel)
        'tiny':   (256, 32, 3),   # 64 tokens, 8 colors per channel = 512 vocab
        'small':  (256, 16, 3),   # 256 tokens, 8 colors per channel = 512 vocab
        'medium': (256, 8, 4),    # 1024 tokens, 16 colors per channel = 4096 vocab
        'large':  (256, 4, 4),    # 4096 tokens, 16 colors per channel = 4096 vocab
    }
    
    def __init__(self, preset='medium'):
        cfg = self.PRESETS[preset]
        self.image_size = cfg[0]
        self.block_size = cfg[1]
        self.bits_per_channel = cfg[2]
        self.preset = preset
        
        self.levels_per_channel = 1 << self.bits_per_channel
        self.vocab_size = self.levels_per_channel ** 3
        self.tokens_per_image = (self.image_size // self.block_size) ** 2
        
        print(f"ColorTokenizer: {preset}")
        print(f"  Block size: {self.block_size}")
        print(f"  Tokens per image: {self.tokens_per_image}")
        print(f"  Levels per channel: {self.levels_per_channel}")
        print(f"  Vocab size: {self.vocab_size}")
    
    def tokenize(self, image: np.ndarray) -> np.ndarray:
        """Tokenize image by quantizing average block colors."""
        if image.shape[:2] != (self.image_size, self.image_size):
            img = Image.fromarray(image)
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
            image = np.array(img)
        
        tokens = []
        bs = self.block_size
        levels = self.levels_per_channel
        
        for y in range(0, self.image_size, bs):
            for x in range(0, self.image_size, bs):
                block = image[y:y+bs, x:x+bs]
                
                # Average color
                avg_color = block.mean(axis=(0, 1))  # [R, G, B]
                
                # Quantize to [0, levels-1]
                r = int(np.clip(avg_color[0] * levels / 256, 0, levels - 1))
                g = int(np.clip(avg_color[1] * levels / 256, 0, levels - 1))
                b = int(np.clip(avg_color[2] * levels / 256, 0, levels - 1))
                
                # Combine into single token
                token = r * levels * levels + g * levels + b
                tokens.append(token)
        
        return np.array(tokens, dtype=np.uint16)
    
    def detokenize(self, tokens: np.ndarray) -> np.ndarray:
        """Reconstruct image from tokens."""
        image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        bs = self.block_size
        levels = self.levels_per_channel
        
        idx = 0
        for y in range(0, self.image_size, bs):
            for x in range(0, self.image_size, bs):
                token = int(tokens[idx])
                
                # Decode token to RGB
                r = token // (levels * levels)
                g = (token // levels) % levels
                b = token % levels
                
                # Scale back to [0, 255]
                r = int((r + 0.5) * 256 / levels)
                g = int((g + 0.5) * 256 / levels)
                b = int((b + 0.5) * 256 / levels)
                
                image[y:y+bs, x:x+bs] = [r, g, b]
                idx += 1
        
        return image
    
    def tokenize_fast(self, image: np.ndarray) -> np.ndarray:
        """Faster tokenization using reshape tricks."""
        if image.shape[:2] != (self.image_size, self.image_size):
            img = Image.fromarray(image)
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
            image = np.array(img)
        
        bs = self.block_size
        n = self.image_size // bs
        levels = self.levels_per_channel
        
        # Reshape to (n, bs, n, bs, 3) then average over block dims
        blocks = image.reshape(n, bs, n, bs, 3).mean(axis=(1, 3))  # (n, n, 3)
        
        # Quantize
        q = (blocks * levels / 256).astype(np.int32)
        q = np.clip(q, 0, levels - 1)
        
        # Combine channels into tokens
        tokens = q[:, :, 0] * levels * levels + q[:, :, 1] * levels + q[:, :, 2]
        
        return tokens.flatten().astype(np.uint16)


def test_roundtrip(image_path: str, preset: str):
    """Test tokenizer."""
    tok = ColorTokenizer(preset)
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize((tok.image_size, tok.image_size), Image.LANCZOS)
    image = np.array(img)
    
    print(f"\nTokenizing {image_path}...")
    tokens = tok.tokenize_fast(image)
    print(f"Tokens: {len(tokens)}, range [{tokens.min()}, {tokens.max()}]")
    print(f"Unique: {len(np.unique(tokens))}")
    print(f"First 20: {tokens[:20]}")
    
    recon = tok.detokenize(tokens)
    
    mse = np.mean((image.astype(float) - recon.astype(float)) ** 2)
    psnr = 10 * np.log10(255**2 / (mse + 1e-10))
    print(f"PSNR: {psnr:.2f} dB")
    
    out_path = Path(image_path).stem + f"_color_{preset}.png"
    Image.fromarray(recon).save(out_path)
    print(f"Saved: {out_path}")


def speed_test(preset: str):
    """Speed test."""
    tok = ColorTokenizer(preset)
    
    images = [np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(100)]
    
    n = 10000
    print(f"\nSpeed test: {n} images...")
    
    start = time.time()
    for i in range(n):
        tok.tokenize_fast(images[i % 100])
    
    elapsed = time.time() - start
    print(f"Rate: {n/elapsed:.0f} img/s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help='Test on image')
    parser.add_argument('--preset', default='medium')
    parser.add_argument('--speed', action='store_true')
    args = parser.parse_args()
    
    if args.test:
        test_roundtrip(args.test, args.preset)
    elif args.speed:
        speed_test(args.preset)
    else:
        # Demo all presets
        for preset in ['tiny', 'small', 'medium', 'large']:
            print(f"\n{'='*50}")
            ColorTokenizer(preset)
        
        speed_test('medium')
