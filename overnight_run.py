#!/usr/bin/env python3
"""
Overnight comprehensive image generation improvement pipeline.

This script:
1. Tokenizes multiple large datasets
2. Builds a comprehensive n-gram index
3. Implements improved sampling strategies
4. Generates many samples for comparison
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from datasets import load_dataset
from flextok.flextok_wrapper import FlexTokFromHub
import torchvision.transforms as T
from collections import Counter
import time
import random

# Configuration
OUTPUT_BASE = Path("/root/imagegram_overnight")
BATCH_SIZE = 48  # Optimized for A40
NUM_WORKERS = 12

# Datasets to use (ordered by quality)
DATASETS = [
    {"name": "food101", "split": "train", "img_key": "image", "expected_size": 75750},
    {"name": "Maysee/tiny-imagenet", "split": "train", "img_key": "image", "expected_size": 100000},
    {"name": "cifar100", "split": "train", "img_key": "img", "expected_size": 50000},
]


class HFImageDataset(Dataset):
    """Wrapper for HuggingFace image datasets."""

    def __init__(self, hf_dataset, transform, img_key="image", label_key="label"):
        self.dataset = hf_dataset
        self.transform = transform
        self.img_key = img_key
        self.label_key = label_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item[self.img_key]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_tensor = self.transform(img)
        label = item.get(self.label_key, -1)
        if isinstance(label, dict):
            label = label.get("fine_label", label.get("label", -1))
        return img_tensor, idx, label


def get_transform():
    """FlexTok expects 256x256, normalized to [-1, 1]."""
    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def load_flextok():
    """Load FlexTok model."""
    print("Loading FlexTok model...")
    model = FlexTokFromHub.from_pretrained("EPFL-VILAB/flextok_d18_d28_dfn")
    model = model.eval().cuda()

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        dummy = torch.randn(BATCH_SIZE, 3, 256, 256).cuda()
        _ = model.tokenize(dummy)
        del dummy
    torch.cuda.empty_cache()

    return model


def tokenize_dataset(model, dataloader, dataset_name, output_dir):
    """Tokenize a dataset and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_tokens = []
    all_labels = []

    print(f"\nTokenizing {dataset_name}...")
    start_time = time.time()

    with torch.no_grad():
        for images, indices, labels in tqdm(dataloader, desc=dataset_name):
            images = images.cuda(non_blocking=True)
            tokens_list = model.tokenize(images)

            for tokens, label in zip(tokens_list, labels):
                token_array = tokens.squeeze(0).cpu().numpy().astype(np.uint32)
                all_tokens.append(token_array)
                all_labels.append(label.item() if hasattr(label, 'item') else int(label))

    elapsed = time.time() - start_time
    print(f"Tokenized {len(all_tokens)} images in {elapsed:.1f}s ({len(all_tokens)/elapsed:.1f} img/s)")

    # Save
    tokens_array = np.stack(all_tokens)
    np.save(output_dir / f"{dataset_name.replace('/', '_')}_tokens.npy", tokens_array)
    np.save(output_dir / f"{dataset_name.replace('/', '_')}_labels.npy", np.array(all_labels))

    return tokens_array, all_labels


def build_combined_index(output_dir, index_dir):
    """Build a combined Fastgram index from all tokenized data."""
    print("\n=== Building Combined Index ===")

    # Load all token files
    all_tokens = []
    all_labels = []

    for token_file in sorted(output_dir.glob("*_tokens.npy")):
        print(f"Loading {token_file.name}...")
        tokens = np.load(token_file)
        all_tokens.append(tokens)

        label_file = token_file.with_name(token_file.name.replace("_tokens", "_labels"))
        if label_file.exists():
            labels = np.load(label_file)
            all_labels.append(labels)

    combined_tokens = np.vstack(all_tokens)
    combined_labels = np.concatenate(all_labels) if all_labels else np.zeros(len(combined_tokens))

    print(f"Combined: {len(combined_tokens)} images, {combined_tokens.shape}")

    # Save combined
    np.save(output_dir / "all_tokens.npy", combined_tokens)
    np.save(output_dir / "all_labels.npy", combined_labels)

    # Create flat tokens with separator
    SEPARATOR = 64000
    flat_tokens = []
    doc_offsets = [0]

    for seq in combined_tokens:
        flat_tokens.extend(seq.tolist())
        flat_tokens.append(SEPARATOR)
        doc_offsets.append(len(flat_tokens))

    flat_array = np.array(flat_tokens, dtype=np.uint32)
    print(f"Flat tokens: {len(flat_array)}")

    # Save for index building
    index_input_dir = output_dir / "index_input"
    index_input_dir.mkdir(exist_ok=True)

    # Convert to u16 (vocab fits)
    flat_u16 = flat_array.astype(np.uint16)
    flat_u16.tofile(index_input_dir / "tokenized.0")

    # Build index using tg_build_index
    import subprocess
    index_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "/root/Fastgram/build/tg_build_index",
        str(index_input_dir),
        str(index_dir),
        "2",  # token_width (u16)
        "4",  # version
        "table_only"
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")

    # Copy tokenized file and create offset file
    import shutil
    shutil.copy(index_input_dir / "tokenized.0", index_dir / "tokenized.0")

    # Create offset file
    offsets = np.array(doc_offsets, dtype=np.uint64)
    offsets.tofile(index_dir / "offset.0")

    # Save config
    config = {
        "vocab_size": 64001,
        "num_tokens_per_image": 256,
        "num_images": len(combined_tokens),
        "separator_token": SEPARATOR,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Index built: {index_dir}")
    return combined_tokens, combined_labels


class ImprovedSampler:
    """Improved n-gram sampling with backoff and smoothing."""

    def __init__(self, engine, tokens_data, labels_data=None):
        self.engine = engine
        self.tokens_data = tokens_data
        self.labels_data = labels_data

        # Precompute statistics
        print("Computing token statistics...")
        self.unigram_counts = Counter(tokens_data.flatten())
        self.first_token_counts = Counter(tokens_data[:, 0])

        # Build unigram distribution
        total = sum(self.unigram_counts.values())
        self.unigram_probs = {t: c/total for t, c in self.unigram_counts.items()}

        # Build first-token distribution
        total_first = sum(self.first_token_counts.values())
        self.first_token_probs = {t: c/total_first for t, c in self.first_token_counts.items()}

        print(f"Unique tokens: {len(self.unigram_counts)}")
        print(f"Unique first tokens: {len(self.first_token_counts)}")

    def sample_token(self, probs_dict, temperature=1.0, top_k=100):
        """Sample from a probability distribution."""
        if not probs_dict:
            return None

        tokens = list(probs_dict.keys())
        probs = np.array([probs_dict[t] for t in tokens])

        # Temperature scaling
        if temperature != 1.0:
            log_probs = np.log(probs + 1e-10)
            log_probs = log_probs / temperature
            probs = np.exp(log_probs)
            probs = probs / probs.sum()

        # Top-k
        if top_k < len(probs):
            top_indices = np.argsort(probs)[-top_k:]
            mask = np.zeros_like(probs)
            mask[top_indices] = 1
            probs = probs * mask
            probs = probs / probs.sum()

        return int(np.random.choice(tokens, p=probs))

    def generate_sequence(self, num_tokens=256, temperature=1.0, top_k=100,
                          label=None, max_context=8):
        """Generate a token sequence with improved backoff."""

        # Start with first token
        if label is not None and self.labels_data is not None:
            # Class-conditional: sample from images of this class
            class_indices = np.where(self.labels_data == label)[0]
            if len(class_indices) > 0:
                first_tokens = self.tokens_data[class_indices, 0]
                first_counts = Counter(first_tokens)
                total = sum(first_counts.values())
                first_probs = {t: c/total for t, c in first_counts.items()}
                first_token = self.sample_token(first_probs, temperature, top_k)
            else:
                first_token = self.sample_token(self.first_token_probs, temperature, top_k)
        else:
            first_token = self.sample_token(self.first_token_probs, temperature, top_k)

        sequence = [first_token]

        # Generate remaining tokens
        while len(sequence) < num_tokens:
            next_token = None

            # Try progressively shorter contexts (backoff)
            for ctx_len in range(min(len(sequence), max_context), 0, -1):
                context = sequence[-ctx_len:]
                result = self.engine.ntd(context)

                if result.get("result_by_token_id"):
                    probs = {int(t): d["prob"] for t, d in result["result_by_token_id"].items()}
                    next_token = self.sample_token(probs, temperature, top_k)
                    if next_token is not None:
                        break

            # Fallback to unigram
            if next_token is None:
                next_token = self.sample_token(self.unigram_probs, temperature, top_k)

            sequence.append(next_token)

        return sequence[:num_tokens]


def generate_samples(model, sampler, output_dir, num_samples=50,
                     temperatures=[0.7, 1.0, 1.3], top_ks=[50, 100, 200]):
    """Generate samples with various settings."""
    print("\n=== Generating Samples ===")

    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    import torchvision.transforms.functional as F

    for temp in temperatures:
        for top_k in top_ks:
            setting_name = f"temp{temp}_topk{top_k}"
            setting_dir = samples_dir / setting_name
            setting_dir.mkdir(exist_ok=True)

            print(f"\nGenerating with temperature={temp}, top_k={top_k}...")

            # Generate token sequences
            token_sequences = []
            for i in range(num_samples):
                seq = sampler.generate_sequence(
                    num_tokens=256,
                    temperature=temp,
                    top_k=top_k
                )
                token_sequences.append(seq)

            # Decode in batches
            all_images = []
            batch_size = 8
            for i in range(0, len(token_sequences), batch_size):
                batch_seqs = token_sequences[i:i+batch_size]
                tokens_list = [
                    torch.tensor(seq, dtype=torch.long).unsqueeze(0).cuda()
                    for seq in batch_seqs
                ]

                with torch.no_grad():
                    images = model.detokenize(
                        tokens_list,
                        timesteps=20,
                        guidance_scale=7.5,
                        perform_norm_guidance=True
                    )
                all_images.extend(images)

            # Save images
            for i, img in enumerate(all_images):
                img = (img + 1) / 2
                img = img.clamp(0, 1)
                img = (img * 255).byte()
                img_pil = F.to_pil_image(img.cpu())
                img_pil.save(setting_dir / f"sample_{i:04d}.png")

            # Save tokens
            np.save(setting_dir / "tokens.npy", np.array(token_sequences))

            print(f"Saved {len(all_images)} samples to {setting_dir}")


def main():
    print("=" * 60)
    print("OVERNIGHT IMAGE GENERATION IMPROVEMENT PIPELINE")
    print("=" * 60)

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Load model once
    model = load_flextok()
    transform = get_transform()

    # Phase 1: Tokenize all datasets
    print("\n" + "=" * 60)
    print("PHASE 1: TOKENIZING DATASETS")
    print("=" * 60)

    for ds_info in DATASETS:
        name = ds_info["name"]
        output_file = OUTPUT_BASE / f"{name.replace('/', '_')}_tokens.npy"

        if output_file.exists():
            print(f"Skipping {name} (already tokenized)")
            continue

        print(f"\nLoading {name}...")
        hf_dataset = load_dataset(name, split=ds_info["split"])
        print(f"Loaded {len(hf_dataset)} images")

        dataset = HFImageDataset(
            hf_dataset,
            transform,
            img_key=ds_info["img_key"],
            label_key="label" if "label" in hf_dataset[0] else "fine_label"
        )

        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=2
        )

        tokenize_dataset(model, dataloader, name, OUTPUT_BASE)
        torch.cuda.empty_cache()

    # Phase 2: Build combined index
    print("\n" + "=" * 60)
    print("PHASE 2: BUILDING INDEX")
    print("=" * 60)

    index_dir = OUTPUT_BASE / "combined_index"
    combined_tokens, combined_labels = build_combined_index(OUTPUT_BASE, index_dir)

    # Phase 3: Initialize improved sampler
    print("\n" + "=" * 60)
    print("PHASE 3: INITIALIZING SAMPLER")
    print("=" * 60)

    from fastgram import GramEngine

    with open(OUTPUT_BASE / "config.json") as f:
        config = json.load(f)

    engine = GramEngine(
        index_dir=str(index_dir),
        eos_token_id=config["separator_token"],
        vocab_size=config["vocab_size"],
        version=4,
        token_dtype="u16"
    )

    sampler = ImprovedSampler(engine, combined_tokens, combined_labels)

    # Phase 4: Generate samples
    print("\n" + "=" * 60)
    print("PHASE 4: GENERATING SAMPLES")
    print("=" * 60)

    generate_samples(
        model, sampler, OUTPUT_BASE,
        num_samples=50,
        temperatures=[0.7, 1.0, 1.3],
        top_ks=[50, 100, 200]
    )

    # Phase 5: Summary
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"Total images tokenized: {len(combined_tokens)}")
    print(f"Index location: {index_dir}")
    print(f"Samples location: {OUTPUT_BASE / 'samples'}")


if __name__ == "__main__":
    main()
