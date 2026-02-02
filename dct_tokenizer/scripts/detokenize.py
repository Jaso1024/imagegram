#!/usr/bin/env python3
"""
Detokenize DCT tokens back to images.
Pure Python implementation matching the C++ CompactTokenizer.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import argparse

# Preset configurations (must match C++)
PRESETS = {
    'tiny': {'image_size': 256, 'block_size': 32, 'num_frequencies': 4, 'quant_bits': 10, 'luma_only': True},
    'small': {'image_size': 256, 'block_size': 32, 'num_frequencies': 4, 'quant_bits': 10, 'luma_only': False},
    'medium': {'image_size': 256, 'block_size': 16, 'num_frequencies': 4, 'quant_bits': 10, 'luma_only': False},
    'large': {'image_size': 256, 'block_size': 8, 'num_frequencies': 16, 'quant_bits': 10, 'luma_only': False},
}

def compute_zigzag(size):
    """Compute zigzag order for a block."""
    zz = []
    for s in range(2 * size - 1):
        if s % 2 == 0:
            for i in range(min(s, size - 1), max(0, s - size + 1) - 1, -1):
                j = s - i
                zz.append(i * size + j)
        else:
            for i in range(max(0, s - size + 1), min(s, size - 1) + 1):
                j = s - i
                zz.append(i * size + j)
    return zz

def compute_dct_basis(size):
    """Compute DCT basis matrix."""
    basis = np.zeros((size, size), dtype=np.float32)
    for k in range(size):
        alpha = np.sqrt(1.0 / size) if k == 0 else np.sqrt(2.0 / size)
        for n in range(size):
            basis[k, n] = alpha * np.cos(np.pi * k * (2 * n + 1) / (2.0 * size))
    return basis

def get_quant_scale(freq_idx, block_size):
    """Get quantization scale for a frequency index."""
    base_scale = 16.0 * (block_size / 8.0)
    freq_factor = 1.0 + freq_idx * 0.5
    return base_scale * freq_factor

def idct_2d(coeffs, basis):
    """2D inverse DCT."""
    size = basis.shape[0]
    # Column IDCT then row IDCT
    temp = basis.T @ coeffs  # Column IDCT
    return temp @ basis      # Row IDCT (transposed)

def detokenize_channel(tokens, config, zigzag, basis):
    """Detokenize a single channel."""
    img_size = config['image_size']
    block_size = config['block_size']
    num_freqs = config['num_frequencies']
    vocab_size = 1 << config['quant_bits']
    
    blocks_per_side = img_size // block_size
    num_blocks = blocks_per_side * blocks_per_side
    
    # Reshape tokens from frequency-first to block-first
    # Input: [freq0_block0, freq0_block1, ..., freq1_block0, ...]
    # Need: coefficients per block
    
    channel = np.zeros((img_size, img_size), dtype=np.float32)
    
    for block_idx in range(num_blocks):
        by = block_idx // blocks_per_side
        bx = block_idx % blocks_per_side
        
        # Gather coefficients for this block
        coeffs = np.zeros(block_size * block_size, dtype=np.float32)
        
        for f in range(num_freqs):
            token_idx = f * num_blocks + block_idx
            if token_idx < len(tokens):
                token = tokens[token_idx]
                scale = get_quant_scale(f, block_size)
                # Dequantize
                coeff = (float(token) - vocab_size / 2.0) * scale
                # Place at zigzag position
                coeffs[zigzag[f]] = coeff
        
        # Reshape to 2D and apply IDCT
        coeffs_2d = coeffs.reshape(block_size, block_size)
        block = idct_2d(coeffs_2d, basis)
        
        # Place block in image
        y_start = by * block_size
        x_start = bx * block_size
        channel[y_start:y_start+block_size, x_start:x_start+block_size] = block
    
    return channel

def detokenize_image(tokens, preset_name):
    """Detokenize tokens to RGB image."""
    config = PRESETS[preset_name]
    
    img_size = config['image_size']
    block_size = config['block_size']
    num_freqs = config['num_frequencies']
    
    blocks_per_side = img_size // block_size
    num_blocks = blocks_per_side * blocks_per_side
    tokens_per_channel = num_blocks * num_freqs
    
    # Precompute
    zigzag = compute_zigzag(block_size)
    basis = compute_dct_basis(block_size)
    
    # Split tokens by channel
    y_tokens = tokens[:tokens_per_channel]
    
    y_channel = detokenize_channel(y_tokens, config, zigzag, basis)
    
    if config['luma_only']:
        # Grayscale
        y_channel = np.clip(y_channel + 128, 0, 255).astype(np.uint8)
        rgb = np.stack([y_channel, y_channel, y_channel], axis=-1)
    else:
        cb_tokens = tokens[tokens_per_channel:2*tokens_per_channel]
        cr_tokens = tokens[2*tokens_per_channel:3*tokens_per_channel]
        
        cb_channel = detokenize_channel(cb_tokens, config, zigzag, basis)
        cr_channel = detokenize_channel(cr_tokens, config, zigzag, basis)
        
        # YCbCr to RGB
        y = y_channel + 128
        r = y + 1.402 * cr_channel
        g = y - 0.344136 * cb_channel - 0.714136 * cr_channel
        b = y + 1.772 * cb_channel
        
        rgb = np.stack([
            np.clip(r, 0, 255).astype(np.uint8),
            np.clip(g, 0, 255).astype(np.uint8),
            np.clip(b, 0, 255).astype(np.uint8),
        ], axis=-1)
    
    return rgb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input tokens file (.npy or .bin)')
    parser.add_argument('output_dir', help='Output directory for images')
    parser.add_argument('--preset', default='small', choices=list(PRESETS.keys()))
    args = parser.parse_args()
    
    config = PRESETS[args.preset]
    tokens_per_image = config['image_size'] // config['block_size']
    tokens_per_image = tokens_per_image ** 2 * config['num_frequencies']
    if not config['luma_only']:
        tokens_per_image *= 3
    
    print(f'Preset: {args.preset}')
    print(f'Tokens per image: {tokens_per_image}')
    
    # Load tokens
    if args.input.endswith('.npy'):
        tokens = np.load(args.input)
        if tokens.ndim == 2:
            # Already split by image
            all_tokens = [tokens[i] for i in range(len(tokens))]
        else:
            # Flat with separators
            sep_idx = np.where(tokens == 65535)[0]
            all_tokens = []
            start = 0
            for end in sep_idx:
                all_tokens.append(tokens[start:end])
                start = end + 1
    else:
        # Binary file
        tokens = np.fromfile(args.input, dtype=np.uint16)
        sep_idx = np.where(tokens == 65535)[0]
        all_tokens = []
        start = 0
        for end in sep_idx:
            all_tokens.append(tokens[start:end])
            start = end + 1
    
    print(f'Found {len(all_tokens)} images')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, img_tokens in enumerate(all_tokens):
        print(f'Decoding image {i+1}/{len(all_tokens)}...', end=' ')
        
        if len(img_tokens) != tokens_per_image:
            print(f'Warning: expected {tokens_per_image} tokens, got {len(img_tokens)}')
            continue
        
        rgb = detokenize_image(img_tokens, args.preset)
        img = Image.fromarray(rgb)
        img.save(output_dir / f'generated_{i:04d}.png')
        print('done')
    
    print(f'\nSaved {len(all_tokens)} images to {output_dir}')

if __name__ == '__main__':
    main()
