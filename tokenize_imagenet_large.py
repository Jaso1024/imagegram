#!/usr/bin/env python3
"""
Tokenize ImageNet 1.28M images with FlexTok.
Expected time: ~12-13 hours at 28 img/s
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from flextok.flextok_wrapper import FlexTokFromHub
import torchvision.transforms as T
import time

# Configuration
OUTPUT_DIR = Path("/root/imagegram_imagenet")
BATCH_SIZE = 64
NUM_WORKERS = 16
CHECKPOINT_EVERY = 50000  # Save checkpoint every 50k images


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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
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

    # Load dataset
    print("Loading ImageNet 256x256...")
    hf_dataset = load_dataset("benjamin-paine/imagenet-1k-256x256", split="train")
    print(f"Total images: {len(hf_dataset)}")

    dataset = ImageNetDataset(hf_dataset, get_transform())
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2
    )

    # Tokenize with checkpointing
    all_tokens = []
    all_labels = []
    start_time = time.time()

    print(f"\nTokenizing {len(dataset)} images...")
    print(f"Estimated time: {len(dataset) / 28 / 3600:.1f} hours")

    with torch.no_grad():
        for batch_idx, (images, indices, labels) in enumerate(tqdm(dataloader, desc="Tokenizing")):
            images = images.cuda(non_blocking=True)
            tokens_list = model.tokenize(images)

            for tokens, label in zip(tokens_list, labels):
                token_array = tokens.squeeze(0).cpu().numpy().astype(np.uint32)
                all_tokens.append(token_array)
                all_labels.append(label.item() if hasattr(label, 'item') else int(label))

            # Checkpoint
            num_done = len(all_tokens)
            if num_done > 0 and num_done % CHECKPOINT_EVERY == 0:
                elapsed = time.time() - start_time
                rate = num_done / elapsed
                remaining = (len(dataset) - num_done) / rate / 3600
                print(f"\nCheckpoint: {num_done} images, {rate:.1f} img/s, ~{remaining:.1f}h remaining")

                # Save checkpoint
                tokens_array = np.stack(all_tokens)
                np.save(OUTPUT_DIR / f"tokens_checkpoint_{num_done}.npy", tokens_array)
                np.save(OUTPUT_DIR / f"labels_checkpoint_{num_done}.npy", np.array(all_labels))

    # Save final
    elapsed = time.time() - start_time
    print(f"\nTokenized {len(all_tokens)} images in {elapsed/3600:.2f} hours ({len(all_tokens)/elapsed:.1f} img/s)")

    tokens_array = np.stack(all_tokens)
    np.save(OUTPUT_DIR / "tokens.npy", tokens_array)
    np.save(OUTPUT_DIR / "labels.npy", np.array(all_labels))

    # Create flat tokens for indexing
    SEPARATOR = 64000
    flat_tokens = []
    for seq in all_tokens:
        flat_tokens.extend(seq.tolist())
        flat_tokens.append(SEPARATOR)

    np.save(OUTPUT_DIR / "tokens_flat.npy", np.array(flat_tokens, dtype=np.uint32))

    # Save config
    config = {
        "vocab_size": 64001,
        "num_tokens_per_image": 256,
        "num_images": len(all_tokens),
        "separator_token": SEPARATOR,
        "dataset": "imagenet-1k-256x256",
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone!")
    print(f"Tokens shape: {tokens_array.shape}")
    print(f"Flat tokens: {len(flat_tokens)}")


if __name__ == "__main__":
    main()
