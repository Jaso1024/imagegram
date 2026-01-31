#!/usr/bin/env python3
"""
Parallel tokenization across multiple GPUs using torch.multiprocessing.
Shards the dataset across GPUs, each GPU processes its shard independently.

Usage:
    python tokenize_parallel.py --num_gpus 8
"""

import argparse
import json
import torch
import torch.multiprocessing as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset
from datasets import load_dataset
from flextok.flextok_wrapper import FlexTokFromHub
import torchvision.transforms as T
import time


# Configuration
OUTPUT_DIR = Path("/root/imagegram_parallel")
BATCH_SIZE = 64
NUM_WORKERS = 4  # Per GPU


class ImageNetDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_tensor = self.transform(img)
        label = item.get("label", -1)
        return img_tensor, idx, label


def get_transform():
    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def tokenize_shard(gpu_id, num_gpus, dataset_size, return_dict):
    """Tokenize a shard of the dataset on a specific GPU."""
    torch.cuda.set_device(gpu_id)

    # Calculate shard indices
    shard_size = dataset_size // num_gpus
    start_idx = gpu_id * shard_size
    end_idx = start_idx + shard_size if gpu_id < num_gpus - 1 else dataset_size

    print(f"[GPU {gpu_id}] Processing indices {start_idx} to {end_idx} ({end_idx - start_idx} images)")

    # Load model on this GPU
    model = FlexTokFromHub.from_pretrained("EPFL-VILAB/flextok_d18_d28_dfn")
    model = model.eval().cuda()

    # Warmup
    with torch.no_grad():
        dummy = torch.randn(BATCH_SIZE, 3, 256, 256).cuda()
        _ = model.tokenize(dummy)
        del dummy
    torch.cuda.empty_cache()

    # Load dataset shard
    hf_dataset = load_dataset("benjamin-paine/imagenet-1k-256x256", split="train")
    full_dataset = ImageNetDataset(hf_dataset, get_transform())

    # Create subset for this shard
    indices = list(range(start_idx, end_idx))
    shard_dataset = Subset(full_dataset, indices)

    dataloader = DataLoader(
        shard_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2
    )

    # Tokenize
    all_tokens = []
    all_labels = []
    all_indices = []

    start_time = time.time()

    with torch.no_grad():
        for images, orig_indices, labels in tqdm(dataloader, desc=f"GPU {gpu_id}", position=gpu_id):
            images = images.cuda(non_blocking=True)
            tokens_list = model.tokenize(images)

            for tokens, orig_idx, label in zip(tokens_list, orig_indices, labels):
                token_array = tokens.squeeze(0).cpu().numpy().astype(np.uint32)
                all_tokens.append(token_array)
                all_indices.append(orig_idx.item())
                all_labels.append(label.item() if hasattr(label, 'item') else int(label))

    elapsed = time.time() - start_time
    rate = len(all_tokens) / elapsed
    print(f"[GPU {gpu_id}] Done: {len(all_tokens)} images in {elapsed:.1f}s ({rate:.1f} img/s)")

    # Save shard results
    tokens_array = np.stack(all_tokens)
    np.save(OUTPUT_DIR / f"tokens_shard_{gpu_id}.npy", tokens_array)
    np.save(OUTPUT_DIR / f"labels_shard_{gpu_id}.npy", np.array(all_labels))
    np.save(OUTPUT_DIR / f"indices_shard_{gpu_id}.npy", np.array(all_indices))

    return_dict[gpu_id] = {
        "num_images": len(all_tokens),
        "elapsed": elapsed,
        "rate": rate
    }


def merge_shards(num_gpus, output_dir):
    """Merge all shard files into a single file."""
    print("\nMerging shards...")

    all_tokens = []
    all_labels = []
    all_indices = []

    for gpu_id in range(num_gpus):
        tokens = np.load(output_dir / f"tokens_shard_{gpu_id}.npy")
        labels = np.load(output_dir / f"labels_shard_{gpu_id}.npy")
        indices = np.load(output_dir / f"indices_shard_{gpu_id}.npy")

        all_tokens.append(tokens)
        all_labels.append(labels)
        all_indices.append(indices)

        print(f"  Loaded shard {gpu_id}: {len(tokens)} images")

    # Concatenate
    tokens_array = np.vstack(all_tokens)
    labels_array = np.concatenate(all_labels)
    indices_array = np.concatenate(all_indices)

    # Sort by original index to maintain order
    sort_order = np.argsort(indices_array)
    tokens_array = tokens_array[sort_order]
    labels_array = labels_array[sort_order]

    # Save merged
    np.save(output_dir / "tokens.npy", tokens_array)
    np.save(output_dir / "labels.npy", labels_array)

    # Create flat tokens for indexing
    SEPARATOR = 64000
    flat_tokens = []
    for seq in tokens_array:
        flat_tokens.extend(seq.tolist())
        flat_tokens.append(SEPARATOR)

    np.save(output_dir / "tokens_flat.npy", np.array(flat_tokens, dtype=np.uint32))

    # Save config
    config = {
        "vocab_size": 64001,
        "num_tokens_per_image": 256,
        "num_images": len(tokens_array),
        "separator_token": SEPARATOR,
        "dataset": "imagenet-1k-256x256",
        "num_gpus": num_gpus,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nMerged: {len(tokens_array)} images")
    print(f"Tokens shape: {tokens_array.shape}")
    print(f"Flat tokens: {len(flat_tokens)}")

    # Cleanup shard files
    for gpu_id in range(num_gpus):
        (output_dir / f"tokens_shard_{gpu_id}.npy").unlink()
        (output_dir / f"labels_shard_{gpu_id}.npy").unlink()
        (output_dir / f"indices_shard_{gpu_id}.npy").unlink()

    return tokens_array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use")
    args = parser.parse_args()

    num_gpus = args.num_gpus
    available_gpus = torch.cuda.device_count()

    if num_gpus > available_gpus:
        print(f"Warning: Requested {num_gpus} GPUs but only {available_gpus} available")
        num_gpus = available_gpus

    print(f"Using {num_gpus} GPUs")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get dataset size
    print("Loading dataset info...")
    hf_dataset = load_dataset("benjamin-paine/imagenet-1k-256x256", split="train")
    dataset_size = len(hf_dataset)
    print(f"Total images: {dataset_size}")
    print(f"Images per GPU: ~{dataset_size // num_gpus}")

    # Start parallel tokenization
    print(f"\nStarting parallel tokenization on {num_gpus} GPUs...")
    start_time = time.time()

    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_dict = manager.dict()

    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=tokenize_shard, args=(gpu_id, num_gpus, dataset_size, return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_elapsed = time.time() - start_time

    # Print stats
    print("\n" + "=" * 60)
    print("TOKENIZATION COMPLETE")
    print("=" * 60)

    total_images = sum(r["num_images"] for r in return_dict.values())
    total_rate = total_images / total_elapsed

    print(f"Total images: {total_images}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Effective rate: {total_rate:.1f} img/s")

    for gpu_id in sorted(return_dict.keys()):
        r = return_dict[gpu_id]
        print(f"  GPU {gpu_id}: {r['num_images']} images, {r['rate']:.1f} img/s")

    # Merge shards
    merge_shards(num_gpus, OUTPUT_DIR)

    print("\nDone!")


if __name__ == "__main__":
    main()
