#!/usr/bin/env python3
"""
Fast tokenization using DataLoader with multiple workers.
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

# Configuration
OUTPUT_DIR = Path("/root/imagegram_data")
BATCH_SIZE = 64
NUM_WORKERS = 8
DATASET_NAME = "cifar10"
DATASET_SPLIT = "test"
MAX_IMAGES = None


class ImageDataset(Dataset):
    def __init__(self, hf_dataset, transform, image_key="img"):
        self.dataset = hf_dataset
        self.transform = transform
        self.image_key = image_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item[self.image_key]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_tensor = self.transform(img)
        label = item.get("label", -1)
        return img_tensor, idx, label


def setup_transform():
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
    print("Warming up model...")
    with torch.no_grad():
        dummy = torch.randn(BATCH_SIZE, 3, 256, 256).cuda()
        _ = model.tokenize(dummy)
        del dummy
    torch.cuda.empty_cache()
    print("Warmup done!")

    # Load dataset
    print(f"Loading {DATASET_NAME}...")
    hf_dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    if MAX_IMAGES:
        hf_dataset = hf_dataset.select(range(MAX_IMAGES))

    image_key = "img" if DATASET_NAME == "cifar10" else "image"
    dataset = ImageDataset(hf_dataset, setup_transform(), image_key)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2
    )
    print(f"Dataset: {len(dataset)} images, {len(dataloader)} batches")

    # Tokenize
    all_tokens = []
    all_indices = []
    all_labels = []

    with torch.no_grad():
        for images, indices, labels in tqdm(dataloader, desc="Tokenizing"):
            images = images.cuda(non_blocking=True)
            tokens_list = model.tokenize(images)

            for tokens, idx, label in zip(tokens_list, indices, labels):
                token_array = tokens.squeeze(0).cpu().numpy().astype(np.uint32)
                all_tokens.append(token_array)
                all_indices.append(idx.item())
                all_labels.append(label.item())

    # Save
    tokens_array = np.stack(all_tokens)
    np.save(OUTPUT_DIR / "tokens.npy", tokens_array)

    metadata = [{"idx": i, "label": l} for i, l in zip(all_indices, all_labels)]
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # Flat tokens with separator
    SEPARATOR = 64000
    flat_tokens = []
    for seq in all_tokens:
        flat_tokens.extend(seq.tolist())
        flat_tokens.append(SEPARATOR)

    np.save(OUTPUT_DIR / "tokens_flat.npy", np.array(flat_tokens, dtype=np.uint32))

    config = {
        "vocab_size": 64001,
        "num_tokens_per_image": 256,
        "num_images": len(all_tokens),
        "separator_token": SEPARATOR,
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone! Saved {len(all_tokens)} images")
    print(f"Token array shape: {tokens_array.shape}")
    print(f"Flat tokens: {len(flat_tokens)}")


if __name__ == "__main__":
    main()
