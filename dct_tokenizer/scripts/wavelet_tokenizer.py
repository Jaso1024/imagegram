#!/usr/bin/env python3
"""
Wavelet-based image tokenizer for n-gram generation.

Wavelets have better spatial locality than DCT:
- Each coefficient = specific location + specific frequency
- Coarse-to-fine natural ordering (LL -> LH/HL/HH at each level)
- Adjacent tokens in raster order are spatially related

Token ordering (coarse to fine):
1. Level N (coarsest) LL coefficients - overall brightness/color  
2. Level N LH, HL, HH - coarse edges
3. Level N-1 LH, HL, HH - medium edges
4. ... down to level 1 (finest details)
"""

import numpy as np
from PIL import Image
import pywt
from pathlib import Path
import argparse
from typing import List, Tuple
import struct
import json

class WaveletTokenizer:
    """Wavelet-based image tokenizer with fixed coefficient counts."""
    
    # Presets: (image_size, wavelet, levels, tokens_per_channel, quant_bits)
    PRESETS = {
        'tiny': (256, 'haar', 5, 85, 10),     # 85*3 = 255 tokens
        'small': (256, 'haar', 4, 256, 10),   # 256*3 = 768 tokens
        'medium': (256, 'haar', 3, 1024, 10), # 1024*3 = 3072 tokens  
        'large': (256, 'haar', 2, 4096, 10),  # 4096*3 = 12288 tokens
    }
    
    def __init__(self, preset='small'):
        if preset not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
            
        self.image_size, self.wavelet, self.levels, self.tokens_per_channel, self.quant_bits = self.PRESETS[preset]
        self.vocab_size = 1 << self.quant_bits
        self.preset = preset
        self.tokens_per_image = self.tokens_per_channel * 3
        
        # Compute structure of wavelet decomposition
        self._compute_structure()
        
        print(f"Wavelet tokenizer: {preset}")
        print(f"  Image size: {self.image_size}")
        print(f"  Wavelet: {self.wavelet}, levels: {self.levels}")
        print(f"  Tokens per channel: {self.tokens_per_channel}")
        print(f"  Tokens per image: {self.tokens_per_image}")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Approx coeffs: {self.approx_size}x{self.approx_size} = {self.approx_count}")
    
    def _compute_structure(self):
        """Compute wavelet coefficient structure."""
        self.approx_size = self.image_size // (2 ** self.levels)
        self.approx_count = self.approx_size ** 2
        
        # Detail bands at each level
        self.detail_sizes = []
        self.detail_counts = []
        size = self.approx_size
        for level in range(self.levels):
            self.detail_sizes.append(size)
            self.detail_counts.append(size * size * 3)  # H, V, D
            size *= 2
    
    def _get_flat_coeffs(self, coeffs):
        """Flatten wavelet coefficients in coarse-to-fine order."""
        flat = []
        
        # Approximation (coarsest)
        cA = coeffs[0].flatten()
        flat.extend(cA)
        
        # Details level by level (coarse to fine)
        for level_idx in range(1, len(coeffs)):
            cH, cV, cD = coeffs[level_idx]
            # Interleave for spatial locality
            h, v, d = cH.flatten(), cV.flatten(), cD.flatten()
            for i in range(len(h)):
                flat.extend([h[i], v[i], d[i]])
        
        return np.array(flat, dtype=np.float32)
    
    def _unflatten_coeffs(self, flat, template_coeffs):
        """Unflatten coefficients back to wavelet structure."""
        idx = 0
        result = []
        
        # Approximation
        size = template_coeffs[0].shape[0]
        count = size * size
        result.append(flat[idx:idx+count].reshape(size, size))
        idx += count
        
        # Details
        for level_idx in range(1, len(template_coeffs)):
            cH, cV, cD = template_coeffs[level_idx]
            size = cH.shape[0]
            count = size * size
            
            h = np.zeros(count)
            v = np.zeros(count)
            d = np.zeros(count)
            
            for i in range(count):
                if idx + 2 < len(flat):
                    h[i] = flat[idx]
                    v[i] = flat[idx + 1]
                    d[i] = flat[idx + 2]
                    idx += 3
            
            result.append((h.reshape(size, size), 
                          v.reshape(size, size), 
                          d.reshape(size, size)))
        
        return result
    
    def tokenize(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Tokenize an image, return tokens and metadata for reconstruction."""
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            img = Image.fromarray(image)
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
            image = np.array(img)
        
        all_tokens = []
        metadata = {'scales': [], 'mins': []}
        
        for c in range(3):
            channel = image[:, :, c].astype(np.float32)
            
            # Wavelet decomposition
            coeffs = pywt.wavedec2(channel, self.wavelet, level=self.levels)
            
            # Flatten in coarse-to-fine order
            flat = self._get_flat_coeffs(coeffs)
            
            # Keep only top coefficients by importance
            if len(flat) > self.tokens_per_channel:
                # For approximation, keep all; for details, sort by magnitude
                approx = flat[:self.approx_count]
                details = flat[self.approx_count:]
                
                # Keep top details by magnitude
                n_details_keep = self.tokens_per_channel - self.approx_count
                if n_details_keep > 0:
                    # Sort by magnitude, keep indices of top
                    importance = np.abs(details)
                    keep_idx = np.argsort(importance)[-n_details_keep:]
                    keep_idx = np.sort(keep_idx)  # Preserve order
                    
                    # Zero out non-kept and keep only up to tokens_per_channel
                    kept_details = np.zeros(n_details_keep)
                    kept_details[:len(keep_idx)] = details[keep_idx]
                    flat = np.concatenate([approx, kept_details])
                else:
                    flat = approx[:self.tokens_per_channel]
            
            # Quantize with global min/max normalization
            vmin, vmax = flat.min(), flat.max()
            scale = vmax - vmin + 1e-8
            
            normalized = (flat - vmin) / scale  # [0, 1]
            tokens = np.clip(normalized * (self.vocab_size - 1), 0, self.vocab_size - 1).astype(np.uint16)
            
            metadata['scales'].append(float(scale))
            metadata['mins'].append(float(vmin))
            
            # Pad if needed
            if len(tokens) < self.tokens_per_channel:
                tokens = np.pad(tokens, (0, self.tokens_per_channel - len(tokens)), 
                               constant_values=self.vocab_size // 2)
            
            all_tokens.extend(tokens[:self.tokens_per_channel])
        
        return np.array(all_tokens, dtype=np.uint16), metadata
    
    def tokenize_simple(self, image: np.ndarray) -> np.ndarray:
        """Tokenize without returning metadata (for batch processing)."""
        tokens, _ = self.tokenize(image)
        return tokens
    
    def detokenize(self, tokens: np.ndarray, metadata: dict = None) -> np.ndarray:
        """Detokenize tokens back to an image."""
        image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Default metadata if not provided
        if metadata is None:
            metadata = {
                'scales': [255.0, 150.0, 150.0],  # Reasonable defaults
                'mins': [0.0, -75.0, -75.0]
            }
        
        for c in range(3):
            channel_tokens = tokens[c * self.tokens_per_channel:(c + 1) * self.tokens_per_channel]
            
            scale = metadata['scales'][c]
            vmin = metadata['mins'][c]
            
            # Dequantize
            normalized = channel_tokens.astype(np.float32) / (self.vocab_size - 1)
            flat = normalized * scale + vmin
            
            # Build full coefficient array (pad with zeros for missing high-freq details)
            total_coeffs = self.approx_count + sum(self.detail_counts)
            full_flat = np.zeros(total_coeffs, dtype=np.float32)
            full_flat[:len(flat)] = flat
            
            # Create template for structure
            dummy = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            template = pywt.wavedec2(dummy, self.wavelet, level=self.levels)
            
            # Unflatten
            coeffs = self._unflatten_coeffs(full_flat, template)
            
            # Reconstruct
            channel = pywt.waverec2(coeffs, self.wavelet)
            channel = channel[:self.image_size, :self.image_size]
            
            image[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
        
        return image


def test_roundtrip(image_path: str, preset: str):
    """Test tokenizer with roundtrip."""
    tokenizer = WaveletTokenizer(preset)
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize((tokenizer.image_size, tokenizer.image_size), Image.LANCZOS)
    image = np.array(img)
    
    print(f"\nOriginal image stats: min={image.min()}, max={image.max()}, mean={image.mean():.1f}")
    
    print(f"\nTokenizing {image_path}...")
    tokens, metadata = tokenizer.tokenize(image)
    print(f"Tokens shape: {tokens.shape}")
    print(f"Token range: [{tokens.min()}, {tokens.max()}]")
    print(f"Unique tokens: {len(np.unique(tokens))}")
    print(f"First 20 tokens: {tokens[:20]}")
    print(f"Metadata: scales={metadata['scales']}, mins={metadata['mins']}")
    
    print("\nDetokenizing...")
    recon = tokenizer.detokenize(tokens, metadata)
    print(f"Recon stats: min={recon.min()}, max={recon.max()}, mean={recon.mean():.1f}")
    
    # Save
    out_path = Path(image_path).stem + f"_wavelet_{preset}_recon.png"
    Image.fromarray(recon).save(out_path)
    print(f"Saved: {out_path}")
    
    # Save original resized for comparison
    Image.fromarray(image).save(Path(image_path).stem + "_original.png")
    
    # Compute PSNR
    mse = np.mean((image.astype(float) - recon.astype(float)) ** 2)
    psnr = 10 * np.log10(255**2 / (mse + 1e-10))
    print(f"PSNR: {psnr:.2f} dB")
    
    return tokens, metadata


def tokenize_batch(input_dir: str, output_file: str, preset: str):
    """Tokenize a batch of images."""
    tokenizer = WaveletTokenizer(preset)
    
    input_path = Path(input_dir)
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']:
        image_paths.extend(input_path.rglob(ext))
    
    print(f"Found {len(image_paths)} images")
    
    with open(output_file, 'wb') as f:
        separator = np.array([65535], dtype=np.uint16)
        
        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert('RGB')
                img = img.resize((tokenizer.image_size, tokenizer.image_size), Image.LANCZOS)
                image = np.array(img)
                
                tokens = tokenizer.tokenize_simple(image)
                f.write(tokens.tobytes())
                f.write(separator.tobytes())
                
                if (i + 1) % 100 == 0:
                    print(f"\r{i+1}/{len(image_paths)}", end='', flush=True)
            except Exception as e:
                print(f"\nError processing {path}: {e}")
    
    # Save config
    config = {
        'preset': preset,
        'tokens_per_image': tokenizer.tokens_per_image,
        'vocab_size': tokenizer.vocab_size,
        'separator_token': 65535,
        'num_images': len(image_paths)
    }
    config_path = str(output_file).rsplit('.', 1)[0] + '_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nSaved to {output_file}")
    print(f"Config: {config_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help='Test roundtrip on an image')
    parser.add_argument('--preset', default='small', choices=WaveletTokenizer.PRESETS.keys())
    parser.add_argument('--batch', help='Input directory for batch processing')
    parser.add_argument('--output', help='Output file for batch processing')
    args = parser.parse_args()
    
    if args.test:
        test_roundtrip(args.test, args.preset)
    elif args.batch:
        if not args.output:
            print("--output required for batch mode")
        else:
            tokenize_batch(args.batch, args.output, args.preset)
    else:
        # Quick test
        print("Testing wavelets with synthetic image...")
        tokenizer = WaveletTokenizer('small')
        
        # Create gradient test image
        x = np.linspace(0, 255, 256)
        test_img = np.zeros((256, 256, 3), dtype=np.uint8)
        test_img[:, :, 0] = x[np.newaxis, :]  # R gradient
        test_img[:, :, 1] = x[:, np.newaxis]  # G gradient
        test_img[:, :, 2] = 128               # B constant
        
        tokens, meta = tokenizer.tokenize(test_img)
        recon = tokenizer.detokenize(tokens, meta)
        
        print(f"\nTokens: {tokens.shape}, range [{tokens.min()}, {tokens.max()}]")
        print(f"Unique: {len(np.unique(tokens))}")
        
        mse = np.mean((test_img.astype(float) - recon.astype(float)) ** 2)
        psnr = 10 * np.log10(255**2 / (mse + 1e-10))
        print(f"Test PSNR: {psnr:.2f} dB")
