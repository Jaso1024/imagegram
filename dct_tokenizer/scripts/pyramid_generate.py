#!/usr/bin/env python3
"""
Generate images using n-gram sampling with pyramid tokens.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import json

def sample_from_ntd(ntd_result, temperature=1.0, top_k=50):
    """Sample from next-token distribution."""
    if not ntd_result.get('result_by_token_id'):
        return None
    
    items = list(ntd_result['result_by_token_id'].items())
    tokens = np.array([int(t) for t, _ in items])
    probs = np.array([d['prob'] for _, d in items])
    
    if temperature != 1.0:
        log_probs = np.log(probs + 1e-10) / temperature
        probs = np.exp(log_probs)
        probs /= probs.sum()
    
    if top_k < len(probs):
        top_idx = np.argsort(probs)[-top_k:]
        mask = np.zeros_like(probs)
        mask[top_idx] = 1
        probs = probs * mask
        probs /= probs.sum()
    
    return int(np.random.choice(tokens, p=probs))


def generate_tokens(engine, num_tokens, temperature=1.0, top_k=50, max_context=8):
    """Generate token sequence."""
    # Start with unigram sampling
    ntd = engine.ntd([])
    first = sample_from_ntd(ntd, temperature, top_k)
    if first is None:
        first = 128  # neutral gray
    
    sequence = [first]
    
    while len(sequence) < num_tokens:
        # Try progressively shorter contexts
        next_token = None
        
        for ctx_len in range(min(len(sequence), max_context), 0, -1):
            context = sequence[-ctx_len:]
            ntd = engine.ntd(context)
            
            if ntd.get('result_by_token_id'):
                next_token = sample_from_ntd(ntd, temperature, top_k)
                if next_token is not None:
                    break
        
        if next_token is None:
            ntd = engine.ntd([])
            next_token = sample_from_ntd(ntd, temperature, top_k)
        
        if next_token is None:
            next_token = 128
        
        sequence.append(next_token)
    
    return np.array(sequence[:num_tokens], dtype=np.uint16)


def pyramid_detokenize(tokens, encoded_levels, output_size):
    """Reconstruct image from pyramid tokens."""
    tokens_per_level = [s * s * 3 for s in encoded_levels]
    
    idx = 0
    base_size = encoded_levels[0]
    base_count = tokens_per_level[0]
    base_tokens = tokens[idx:idx + base_count]
    idx += base_count
    
    current = base_tokens.reshape(base_size, base_size, 3).astype(np.float32)
    
    for i in range(1, len(encoded_levels)):
        curr_size = encoded_levels[i]
        level_count = tokens_per_level[i]
        
        img_pil = Image.fromarray(np.clip(current, 0, 255).astype(np.uint8))
        upsampled = np.array(img_pil.resize((curr_size, curr_size), Image.BILINEAR)).astype(np.float32)
        
        residual_tokens = tokens[idx:idx + level_count]
        idx += level_count
        residual = residual_tokens.reshape(curr_size, curr_size, 3).astype(np.float32) - 128
        
        current = upsampled + residual
    
    final_size = encoded_levels[-1]
    if final_size != output_size:
        img_pil = Image.fromarray(np.clip(current, 0, 255).astype(np.uint8))
        current = np.array(img_pil.resize((output_size, output_size), Image.BILINEAR))
    
    return np.clip(current, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index-dir', default='/root/pyramid_index')
    parser.add_argument('--config', default='/root/pyramid_food101_config.json')
    parser.add_argument('--output-dir', default='/root/pyramid_generated')
    parser.add_argument('--num-images', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--max-context', type=int, default=16)
    args = parser.parse_args()
    
    from fastgram import GramEngine
    
    with open(args.config) as f:
        config = json.load(f)
    
    tokens_per_image = config['tokens_per_image']
    encoded_levels = config['encoded_levels']
    output_size = config['output_size']
    
    print(f'Loading index from {args.index_dir}...')
    engine = GramEngine(
        index_dir=args.index_dir,
        eos_token_id=config['separator_token'],
        vocab_size=config['vocab_size'],
        version=4,
        token_dtype='u16'
    )
    print('Loaded!')
    
    print(f'\nGenerating {args.num_images} images...')
    print(f'  Tokens per image: {tokens_per_image}')
    print(f'  Temperature: {args.temperature}')
    print(f'  Top-k: {args.top_k}')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(args.num_images):
        print(f'  Generating {i+1}/{args.num_images}...', end=' ')
        
        tokens = generate_tokens(
            engine,
            tokens_per_image,
            temperature=args.temperature,
            top_k=args.top_k,
            max_context=args.max_context
        )
        
        image = pyramid_detokenize(tokens, encoded_levels, output_size)
        
        out_path = output_dir / f'generated_{i:04d}.png'
        Image.fromarray(image).save(out_path)
        print(f'saved {out_path.name}')
    
    print(f'\nDone! Images saved to {output_dir}')


if __name__ == '__main__':
    main()
