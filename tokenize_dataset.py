#!/usr/bin/env python3
"""
Tokenize an image dataset using FlexTok and save tokens for Fastgram indexing.
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from flextok.flextok_wrapper import FlexTokFromHub
import torchvision.transforms as T

# Configuration
OUTPUT_DIR = Path("/root/imagegram_data")
BATCH_SIZE = 32  # Adjust based on GPU memory
NUM_TOKENS = 256  # Full sequence length
DATASET_NAME = "cifar10"  # Start with CIFAR-10 for testing
DATASET_SPLIT = "test"  # 10k images
MAX_IMAGES = None  # Set to a number to limit for testing, None for all

# Dataset key mapping (different datasets use different keys)
IMAGE_KEYS = {"cifar10": "img", "imagenet-1k": "image", "ILSVRC/imagenet-1k": "image"}

def setup_model():
    """Load FlexTok model."""
    print("Loading FlexTok model...")
    model = FlexTokFromHub.from_pretrained("EPFL-VILAB/flextok_d18_d28_dfn")
    model = model.eval().cuda()
    print("Model loaded!")
    return model

def setup_transform():
    """Create image transform for FlexTok (256x256, normalized to [-1, 1])."""
    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
    ])

def load_images_batch(dataset, indices, transform, image_key):
    """Load a batch of images from dataset."""
    images = []
    valid_indices = []
    for idx in indices:
        try:
            img = dataset[idx][image_key]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_tensor = transform(img)
            images.append(img_tensor)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error loading image {idx}: {e}")

    if images:
        return torch.stack(images), valid_indices
    return None, []

def tokenize_dataset(model, dataset, transform, output_dir, image_key):
    """Tokenize all images and save tokens."""
    output_dir.mkdir(parents=True, exist_ok=True)

    n_images = len(dataset) if MAX_IMAGES is None else min(MAX_IMAGES, len(dataset))
    all_tokens = []
    metadata = []

    print(f"Tokenizing {n_images} images...")

    with torch.no_grad():
        for batch_start in tqdm(range(0, n_images, BATCH_SIZE)):
            batch_end = min(batch_start + BATCH_SIZE, n_images)
            indices = list(range(batch_start, batch_end))

            images, valid_indices = load_images_batch(dataset, indices, transform, image_key)
            if images is None:
                continue

            images = images.cuda()

            # Tokenize - returns list of token tensors with shape [1, 256]
            tokens_list = model.tokenize(images)

            for tokens, idx in zip(tokens_list, valid_indices):
                # tokens has shape [1, 256], squeeze to [256]
                token_array = tokens.squeeze(0).cpu().numpy().astype(np.uint32)
                all_tokens.append(token_array)

                # Store metadata
                meta = {
                    "idx": idx,
                    "label": dataset[idx].get("label", -1),
                }
                metadata.append(meta)

    # Save tokens as a single file (for Fastgram indexing)
    tokens_array = np.array(all_tokens, dtype=np.uint32)
    np.save(output_dir / "tokens.npy", tokens_array)

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # Also save as flat sequence for Fastgram (with separator tokens)
    # Using token 64000 as document separator (outside vocab)
    SEPARATOR = 64000
    flat_tokens = []
    for token_seq in all_tokens:
        flat_tokens.extend(token_seq.tolist())
        flat_tokens.append(SEPARATOR)

    flat_array = np.array(flat_tokens, dtype=np.uint32)
    np.save(output_dir / "tokens_flat.npy", flat_array)

    print(f"Saved {len(all_tokens)} tokenized images")
    print(f"Token array shape: {tokens_array.shape}")
    print(f"Flat tokens length: {len(flat_array)}")

    # Save config for Fastgram
    config = {
        "vocab_size": 64001,  # 64000 + separator
        "num_tokens_per_image": NUM_TOKENS,
        "num_images": len(all_tokens),
        "separator_token": SEPARATOR,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    return tokens_array

def main():
    # Setup
    model = setup_model()
    transform = setup_transform()

    # Get image key for dataset
    image_key = IMAGE_KEYS.get(DATASET_NAME, "image")
    print(f"Using image key: {image_key}")

    # Load dataset
    print(f"Loading dataset: {DATASET_NAME} ({DATASET_SPLIT})...")

    if DATASET_NAME == "cifar10":
        dataset = load_dataset("cifar10", split=DATASET_SPLIT)
    else:
        # ImageNet requires authentication
        dataset = load_dataset(
            DATASET_NAME,
            split=DATASET_SPLIT,
            trust_remote_code=True
        )

    print(f"Dataset loaded: {len(dataset)} images")

    # Tokenize
    tokens = tokenize_dataset(model, dataset, transform, OUTPUT_DIR, image_key)

    print("Done!")
    return tokens

if __name__ == "__main__":
    main()
