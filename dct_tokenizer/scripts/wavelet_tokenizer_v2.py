#!/usr/bin/env python3
"""
Wavelet tokenizer v2 - Fixed quantization without metadata.

Key insight: Use separate fixed ranges for approximation vs detail coefficients:
- Approx coeffs: [0, 255] (pixel values averaged)
- Detail coeffs: [-128, 128] (edge responses)

This allows reconstruction without per-image metadata.
"""

import numpy as np
from PIL import Image
import pywt
from pathlib import Path
import argparse
import json
import time

class WaveletTokenizerV2:
    """Wavelet tokenizer with fixed quantization ranges."""
    
    PRESETS = {
        # (image_size, wavelet, levels, detail_levels_to_keep, quant_bits)
        # Total = 3 * (approx_size^2 + detail_coeffs)
        'tiny':   (256, 'haar', 6, 0, 10),   # 3 * 16 = 48 tokens (approx only)
        'small':  (256, 'haar', 6, 1, 10),   # 3 * (16 + 48) = 192 tokens  
        'medium': (256, 'haar', 5, 1, 10),   # 3 * (64 + 192) = 768 tokens
        'large':  (256, 'haar', 4, 2, 10),   # 3 * (256 + 768 + 3072) = 12288 tokens
    }
    
    def __init__(self, preset='small'):
        cfg = self.PRESETS[preset]
        self.image_size = cfg[0]
        self.wavelet = cfg[1]
        self.levels = cfg[2]
        self.detail_levels = cfg[3]
        self.quant_bits = cfg[4]
        self.vocab_size = 1 << self.quant_bits
        self.preset = preset
        
        # Compute token counts
        self.approx_size = self.image_size // (2 ** self.levels)
        self.approx_count = self.approx_size ** 2
        
        # Detail counts per level (each level has 3 subbands: H, V, D)
        self.detail_counts = []
        size = self.approx_size
        for _ in range(self.detail_levels):
            self.detail_counts.append(size * size * 3)
            size *= 2
        
        self.total_detail = sum(self.detail_counts)
        self.tokens_per_channel = self.approx_count + self.total_detail
        self.tokens_per_image = self.tokens_per_channel * 3
        
        # Fixed quantization ranges - account for haar wavelet scaling
        # Haar wavelet sums values, so approx coeffs are scaled by 2^levels
        scale_factor = 2 ** self.levels
        self.approx_range = (0, 255 * scale_factor)
        self.detail_range = (-300 * scale_factor / 4, 300 * scale_factor / 4)  # Detail scaling is less
        
        print(f"WaveletTokenizer v2: {preset}")
        print(f"  Levels: {self.levels}, detail_levels: {self.detail_levels}")
        print(f"  Approx: {self.approx_size}x{self.approx_size} = {self.approx_count}")
        print(f"  Details: {self.detail_counts} = {self.total_detail}")
        print(f"  Tokens/channel: {self.tokens_per_channel}, total: {self.tokens_per_image}")
        print(f"  Vocab: {self.vocab_size}")
    
    def _quantize_approx(self, coeffs):
        """Quantize approximation coefficients (pixel-like values)."""
        lo, hi = self.approx_range
        normalized = (coeffs - lo) / (hi - lo + 1e-8)
        tokens = np.clip(normalized * (self.vocab_size - 1), 0, self.vocab_size - 1)
        return tokens.astype(np.uint16)
    
    def _dequantize_approx(self, tokens):
        """Dequantize approximation tokens."""
        lo, hi = self.approx_range
        normalized = tokens.astype(np.float32) / (self.vocab_size - 1)
        return normalized * (hi - lo) + lo
    
    def _quantize_detail(self, coeffs):
        """Quantize detail coefficients (edge responses)."""
        lo, hi = self.detail_range
        # Clip to range first
        coeffs = np.clip(coeffs, lo, hi)
        normalized = (coeffs - lo) / (hi - lo + 1e-8)
        tokens = np.clip(normalized * (self.vocab_size - 1), 0, self.vocab_size - 1)
        return tokens.astype(np.uint16)
    
    def _dequantize_detail(self, tokens):
        """Dequantize detail tokens."""
        lo, hi = self.detail_range
        normalized = tokens.astype(np.float32) / (self.vocab_size - 1)
        return normalized * (hi - lo) + lo
    
    def tokenize(self, image: np.ndarray) -> np.ndarray:
        """Tokenize image to wavelet tokens."""
        if image.shape[:2] != (self.image_size, self.image_size):
            img = Image.fromarray(image)
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
            image = np.array(img)
        
        all_tokens = []
        
        for c in range(3):
            channel = image[:, :, c].astype(np.float32)
            coeffs = pywt.wavedec2(channel, self.wavelet, level=self.levels)
            
            # Approximation
            approx = coeffs[0]
            approx_tokens = self._quantize_approx(approx.flatten())
            all_tokens.extend(approx_tokens)
            
            # Details (only keep detail_levels levels, from coarse to fine)
            for level_idx in range(1, self.detail_levels + 1):
                cH, cV, cD = coeffs[level_idx]
                # Interleave H, V, D for each position
                h, v, d = cH.flatten(), cV.flatten(), cD.flatten()
                detail = np.stack([h, v, d], axis=1).flatten()
                detail_tokens = self._quantize_detail(detail)
                all_tokens.extend(detail_tokens)
        
        return np.array(all_tokens, dtype=np.uint16)
    
    def detokenize(self, tokens: np.ndarray) -> np.ndarray:
        """Detokenize tokens back to image."""
        image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        for c in range(3):
            start = c * self.tokens_per_channel
            channel_tokens = tokens[start:start + self.tokens_per_channel]
            
            # Build coefficient structure
            dummy = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            template = pywt.wavedec2(dummy, self.wavelet, level=self.levels)
            
            idx = 0
            
            # Approximation
            approx_tokens = channel_tokens[idx:idx + self.approx_count]
            approx = self._dequantize_approx(approx_tokens).reshape(self.approx_size, self.approx_size)
            template[0] = approx
            idx += self.approx_count
            
            # Details
            for level_idx in range(1, self.detail_levels + 1):
                size = template[level_idx][0].shape[0]
                count = size * size
                
                detail_tokens = channel_tokens[idx:idx + count * 3]
                if len(detail_tokens) == count * 3:
                    detail = self._dequantize_detail(detail_tokens)
                    # De-interleave H, V, D
                    detail = detail.reshape(-1, 3)
                    h, v, d = detail[:, 0], detail[:, 1], detail[:, 2]
                    template[level_idx] = (
                        h.reshape(size, size),
                        v.reshape(size, size),
                        d.reshape(size, size)
                    )
                idx += count * 3
            
            # Zero out remaining levels (fine details we didn't encode)
            for level_idx in range(self.detail_levels + 1, len(template)):
                size = template[level_idx][0].shape[0]
                template[level_idx] = (
                    np.zeros((size, size)),
                    np.zeros((size, size)),
                    np.zeros((size, size))
                )
            
            # Reconstruct
            channel = pywt.waverec2(template, self.wavelet)
            channel = channel[:self.image_size, :self.image_size]
            image[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
        
        return image


def test_roundtrip(image_path: str, preset: str):
    """Test roundtrip quality."""
    tokenizer = WaveletTokenizerV2(preset)
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize((tokenizer.image_size, tokenizer.image_size), Image.LANCZOS)
    image = np.array(img)
    
    print(f"\nTokenizing {image_path}...")
    tokens = tokenizer.tokenize(image)
    print(f"Tokens: {len(tokens)}, range [{tokens.min()}, {tokens.max()}]")
    print(f"Unique: {len(np.unique(tokens))}")
    print(f"First 30: {tokens[:30]}")
    
    print("\nDetokenizing...")
    recon = tokenizer.detokenize(tokens)
    
    # PSNR
    mse = np.mean((image.astype(float) - recon.astype(float)) ** 2)
    psnr = 10 * np.log10(255**2 / (mse + 1e-10))
    print(f"PSNR: {psnr:.2f} dB")
    
    # Save
    out_path = Path(image_path).stem + f"_wavelet_v2_{preset}.png"
    Image.fromarray(recon).save(out_path)
    print(f"Saved: {out_path}")
    
    return tokens


def speed_test(preset: str, n_images: int = 1000):
    """Test tokenization speed."""
    tokenizer = WaveletTokenizerV2(preset)
    
    # Create random images
    images = [np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(100)]
    
    print(f"\nSpeed test: {n_images} images...")
    start = time.time()
    
    for i in range(n_images):
        img = images[i % 100]
        tokens = tokenizer.tokenize(img)
    
    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s")
    print(f"Rate: {n_images / elapsed:.0f} img/s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help='Test on image')
    parser.add_argument('--preset', default='small')
    parser.add_argument('--speed', action='store_true', help='Run speed test')
    args = parser.parse_args()
    
    if args.test:
        test_roundtrip(args.test, args.preset)
    elif args.speed:
        speed_test(args.preset)
    else:
        # Quick demo
        for preset in ['tiny', 'small', 'medium', 'large']:
            print(f"\n{'='*50}")
            tokenizer = WaveletTokenizerV2(preset)
