#!/usr/bin/env python3
"""
Generate images using n-gram sampling from Fastgram + FlexTok decoding.
"""

import json
import torch
import numpy as np
from pathlib import Path
from fastgram import GramEngine
from flextok.flextok_wrapper import FlexTokFromHub
import random

# Configuration
INDEX_DIR = "/root/imagegram_index"
DATA_DIR = Path("/root/imagegram_data")
OUTPUT_DIR = Path("/root/generated_images")
NUM_TOKENS = 256
NUM_IMAGES = 8  # Number of images to generate
TEMPERATURE = 1.0  # Sampling temperature
TOP_K = 50  # Only sample from top-k tokens


def load_engine():
    """Load Fastgram engine."""
    with open(DATA_DIR / "config.json") as f:
        config = json.load(f)

    engine = GramEngine(
        index_dir=INDEX_DIR,
        eos_token_id=config["separator_token"],
        vocab_size=config["vocab_size"],
        version=4,
        token_dtype="u16"
    )
    return engine, config


def load_flextok():
    """Load FlexTok model for decoding."""
    print("Loading FlexTok...")
    model = FlexTokFromHub.from_pretrained("EPFL-VILAB/flextok_d18_d28_dfn")
    model = model.eval().cuda()
    return model


def sample_from_distribution(result, temperature=1.0, top_k=50):
    """Sample a token from the next-token distribution."""
    if not result.get("result_by_token_id"):
        return None

    items = list(result["result_by_token_id"].items())
    tokens = [int(t) for t, _ in items]
    probs = np.array([d["prob"] for _, d in items])

    # Apply temperature
    if temperature != 1.0:
        log_probs = np.log(probs + 1e-10)
        log_probs = log_probs / temperature
        probs = np.exp(log_probs)
        probs = probs / probs.sum()

    # Top-k filtering
    if top_k < len(probs):
        top_indices = np.argsort(probs)[-top_k:]
        mask = np.zeros_like(probs)
        mask[top_indices] = 1
        probs = probs * mask
        probs = probs / probs.sum()

    # Sample
    sampled_idx = np.random.choice(len(tokens), p=probs)
    return tokens[sampled_idx]


def generate_token_sequence(engine, num_tokens=256, temperature=1.0, top_k=50, seed_tokens=None):
    """Generate a token sequence using n-gram sampling."""
    if seed_tokens is None:
        # Start with empty context - sample from unigram
        result = engine.ntd([])
        if not result.get("result_by_token_id"):
            # Fallback: use a random common first token
            tokens_data = np.load(DATA_DIR / "tokens.npy")
            first_token = int(random.choice(tokens_data[:, 0]))
            sequence = [first_token]
        else:
            first_token = sample_from_distribution(result, temperature, top_k)
            if first_token is None:
                tokens_data = np.load(DATA_DIR / "tokens.npy")
                first_token = int(random.choice(tokens_data[:, 0]))
            sequence = [first_token]
    else:
        sequence = list(seed_tokens)

    # Generate remaining tokens
    while len(sequence) < num_tokens:
        # Use last N tokens as context (N = suffix_len from engine)
        # Try progressively shorter contexts if no matches
        found = False
        for ctx_len in range(min(len(sequence), 10), 0, -1):
            context = sequence[-ctx_len:]
            result = engine.ntd(context)

            if result.get("result_by_token_id"):
                next_token = sample_from_distribution(result, temperature, top_k)
                if next_token is not None:
                    sequence.append(next_token)
                    found = True
                    break

        if not found:
            # No n-gram match found - sample from unigram distribution
            result = engine.ntd([])
            if result.get("result_by_token_id"):
                next_token = sample_from_distribution(result, temperature, top_k)
                if next_token:
                    sequence.append(next_token)
            else:
                # Complete fallback - random token from training data
                tokens_data = np.load(DATA_DIR / "tokens.npy")
                next_token = int(random.choice(tokens_data.flatten()))
                sequence.append(next_token)

    return sequence[:num_tokens]


def decode_tokens(model, token_sequences, timesteps=20, guidance_scale=7.5):
    """Decode token sequences to images using FlexTok."""
    # Convert to tensor format expected by FlexTok
    # FlexTok expects list of tensors with shape [1, num_tokens]
    tokens_list = [
        torch.tensor(seq, dtype=torch.long).unsqueeze(0).cuda()
        for seq in token_sequences
    ]

    with torch.no_grad():
        images = model.detokenize(
            tokens_list,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            perform_norm_guidance=True
        )

    return images


def save_images(images, output_dir, prefix="generated"):
    """Save generated images."""
    import torchvision.transforms.functional as F
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(images):
        # Convert from [-1, 1] to [0, 255]
        img = (img + 1) / 2
        img = img.clamp(0, 1)
        img = (img * 255).byte()

        # Convert to PIL
        img_pil = F.to_pil_image(img.cpu())
        img_pil.save(output_dir / f"{prefix}_{i:04d}.png")


def main():
    print("=== N-gram Image Generation ===")

    # Setup
    engine, config = load_engine()
    model = load_flextok()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {NUM_IMAGES} images...")
    print(f"Temperature: {TEMPERATURE}, Top-k: {TOP_K}")

    # Generate token sequences
    token_sequences = []
    for i in range(NUM_IMAGES):
        print(f"Generating sequence {i+1}/{NUM_IMAGES}...", end=" ")
        seq = generate_token_sequence(
            engine,
            num_tokens=NUM_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K
        )
        token_sequences.append(seq)
        print(f"Done. First 5 tokens: {seq[:5]}")

    # Decode to images
    print("\nDecoding tokens to images...")
    images = decode_tokens(model, token_sequences)

    # Save
    print(f"Saving to {OUTPUT_DIR}...")
    save_images(images, OUTPUT_DIR)

    print(f"\nDone! Generated {NUM_IMAGES} images.")

    # Also save token sequences for analysis
    np.save(OUTPUT_DIR / "generated_tokens.npy", np.array(token_sequences))


if __name__ == "__main__":
    main()
