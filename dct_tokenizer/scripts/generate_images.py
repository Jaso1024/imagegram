#!/usr/bin/env python3
"""
Generate images using n-gram sampling from DCT tokens.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

def sample_from_ntd(ntd_result, temperature=1.0, top_k=50):
    """Sample a token from next-token distribution."""
    if not ntd_result.get('result_by_token_id'):
        return None
    
    items = list(ntd_result['result_by_token_id'].items())
    tokens = np.array([int(t) for t, _ in items])
    probs = np.array([d['prob'] for _, d in items])
    
    # Temperature
    if temperature != 1.0:
        log_probs = np.log(probs + 1e-10) / temperature
        probs = np.exp(log_probs)
        probs /= probs.sum()
    
    # Top-k
    if top_k < len(probs):
        top_idx = np.argsort(probs)[-top_k:]
        mask = np.zeros_like(probs)
        mask[top_idx] = 1
        probs = probs * mask
        probs /= probs.sum()
    
    return int(np.random.choice(tokens, p=probs))

def generate_sequence(engine, num_tokens, temperature=1.0, top_k=50, max_context=10):
    """Generate a token sequence using n-gram sampling."""
    # Start with empty context - sample first token
    ntd = engine.ntd([])
    first_token = sample_from_ntd(ntd, temperature, top_k)
    
    if first_token is None:
        # Fallback: use a common starting token
        first_token = 576  # Common DC coefficient value
    
    sequence = [first_token]
    
    # Generate remaining tokens
    while len(sequence) < num_tokens:
        # Try progressively shorter contexts (backoff)
        next_token = None
        
        for ctx_len in range(min(len(sequence), max_context), 0, -1):
            context = sequence[-ctx_len:]
            ntd = engine.ntd(context)
            
            if ntd.get('result_by_token_id'):
                next_token = sample_from_ntd(ntd, temperature, top_k)
                if next_token is not None:
                    break
        
        if next_token is None:
            # Fallback to unigram
            ntd = engine.ntd([])
            next_token = sample_from_ntd(ntd, temperature, top_k)
        
        if next_token is None:
            next_token = 512  # Fallback to middle value
        
        sequence.append(next_token)
    
    return sequence[:num_tokens]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index-dir', default='/root/imagenet_index')
    parser.add_argument('--config', default='/root/index_input/config.json')
    parser.add_argument('--output-dir', default='/root/generated_images')
    parser.add_argument('--num-images', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--max-context', type=int, default=8)
    parser.add_argument('--preset', default='small', help='DCT preset used for tokenization')
    args = parser.parse_args()
    
    from fastgram import GramEngine
    
    # Load config
    with open(args.config) as f:
        config = json.load(f)
    
    tokens_per_image = config['tokens_per_image']
    
    print(f'Loading index from {args.index_dir}...')
    engine = GramEngine(
        index_dir=args.index_dir,
        eos_token_id=config['separator_token'],
        vocab_size=config['vocab_size'],
        version=4,
        token_dtype='u16'
    )
    print('Index loaded!')
    
    print(f'\nGenerating {args.num_images} images...')
    print(f'  Tokens per image: {tokens_per_image}')
    print(f'  Temperature: {args.temperature}')
    print(f'  Top-k: {args.top_k}')
    print(f'  Max context: {args.max_context}')
    
    # Generate token sequences
    all_tokens = []
    for i in range(args.num_images):
        print(f'  Generating image {i+1}/{args.num_images}...', end=' ')
        seq = generate_sequence(
            engine,
            tokens_per_image,
            temperature=args.temperature,
            top_k=args.top_k,
            max_context=args.max_context
        )
        all_tokens.append(seq)
        print(f'done (first 5 tokens: {seq[:5]})')
    
    # Save tokens
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tokens_array = np.array(all_tokens, dtype=np.uint16)
    np.save(output_dir / 'generated_tokens.npy', tokens_array)
    print(f'\nSaved tokens to {output_dir / "generated_tokens.npy"}')
    
    # Decode to images using DCT detokenizer
    print('\nDecoding tokens to images...')
    
    # Write tokens as flat binary for the detokenizer
    flat_tokens = []
    for seq in all_tokens:
        flat_tokens.extend(seq)
        flat_tokens.append(65535)  # separator
    
    flat_path = output_dir / 'tokens_flat.bin'
    np.array(flat_tokens, dtype=np.uint16).tofile(flat_path)
    
    print(f'Run detokenizer manually:')
    print(f'  /root/dct_tokenizer/build/tokenize_compact --detokenize {args.preset} {flat_path} {output_dir}')

if __name__ == '__main__':
    main()
