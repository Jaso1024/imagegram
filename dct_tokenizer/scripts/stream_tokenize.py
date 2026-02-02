#!/usr/bin/env python3
"""
Stream images from HuggingFace datasets and tokenize them on-the-fly.
Outputs tokens directly to a flat binary file without storing images.

This is memory-efficient and disk-efficient for large datasets.
"""

import argparse
import subprocess
import tempfile
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Thread
import struct

def get_dataset_info(dataset_name, split="train"):
    """Get dataset info from HuggingFace."""
    from datasets import load_dataset_builder
    builder = load_dataset_builder(dataset_name)
    info = builder.info
    return {
        "num_examples": info.splits[split].num_examples if split in info.splits else None,
        "features": str(info.features)
    }

def stream_tokenize(
    dataset_name: str,
    output_file: str,
    preset: str = "small",
    split: str = "train",
    max_images: int = None,
    batch_size: int = 100,
    num_workers: int = 8,
    tokenizer_path: str = "./build/tokenize_compact",
    image_key: str = "image"
):
    """Stream images and tokenize them."""
    from datasets import load_dataset
    import io
    from PIL import Image
    
    print(f"Loading dataset: {dataset_name}, split: {split}")
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    
    # Determine tokens per image based on preset
    tokens_per_image = {
        "tiny": 256,
        "small": 768,
        "medium": 3072,
        "large": 12288
    }[preset]
    
    separator = 65535
    
    print(f"Preset: {preset}, tokens/image: {tokens_per_image}")
    print(f"Output: {output_file}")
    
    # Open output file
    out_f = open(output_file, "wb")
    
    processed = 0
    failed = 0
    
    # Create temp directory for batch processing
    with tempfile.TemporaryDirectory() as tmpdir:
        batch_dir = Path(tmpdir) / "batch"
        batch_dir.mkdir()
        
        batch_images = []
        batch_idx = 0
        
        import time
        start_time = time.time()
        
        for item in dataset:
            if max_images and processed >= max_images:
                break
            
            try:
                img = item[image_key]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Save to temp file
                img_path = batch_dir / f"{batch_idx:06d}.png"
                img.save(img_path, format="PNG")
                batch_images.append(img_path)
                batch_idx += 1
                
                # Process batch
                if len(batch_images) >= batch_size:
                    # Run tokenizer
                    token_file = Path(tmpdir) / "tokens.bin"
                    cmd = [tokenizer_path, preset, str(batch_dir), str(token_file)]
                    result = subprocess.run(cmd, capture_output=True)
                    
                    if result.returncode == 0 and token_file.exists():
                        # Read tokens and write to output
                        with open(token_file, "rb") as tf:
                            tokens_data = tf.read()
                        out_f.write(tokens_data)
                        processed += len(batch_images)
                    else:
                        failed += len(batch_images)
                        print(f"\nTokenizer error: {result.stderr.decode()}")
                    
                    # Clear batch
                    for p in batch_images:
                        p.unlink()
                    batch_images = []
                    batch_idx = 0
                    
                    # Progress
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    print(f"\rProcessed: {processed}, Failed: {failed}, Rate: {rate:.1f} img/s", end="")
                    
            except Exception as e:
                failed += 1
                print(f"\nError processing image: {e}")
        
        # Process remaining
        if batch_images:
            token_file = Path(tmpdir) / "tokens.bin"
            cmd = [tokenizer_path, preset, str(batch_dir), str(token_file)]
            result = subprocess.run(cmd, capture_output=True)
            
            if result.returncode == 0 and token_file.exists():
                with open(token_file, "rb") as tf:
                    tokens_data = tf.read()
                out_f.write(tokens_data)
                processed += len(batch_images)
    
    out_f.close()
    
    elapsed = time.time() - start_time
    print(f"\n\n=== Summary ===")
    print(f"Processed: {processed} images")
    print(f"Failed: {failed} images")
    print(f"Time: {elapsed:.1f}s")
    print(f"Rate: {processed/elapsed:.1f} img/s")
    print(f"Output: {output_file}")
    print(f"Output size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")

def main():
    parser = argparse.ArgumentParser(description="Stream tokenize images from HuggingFace")
    parser.add_argument("dataset", help="HuggingFace dataset name")
    parser.add_argument("output", help="Output token file (.bin)")
    parser.add_argument("--preset", default="small", choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--tokenizer", default="./build/tokenize_compact")
    parser.add_argument("--image-key", default="image")
    
    args = parser.parse_args()
    
    stream_tokenize(
        args.dataset,
        args.output,
        preset=args.preset,
        split=args.split,
        max_images=args.max_images,
        batch_size=args.batch_size,
        tokenizer_path=args.tokenizer,
        image_key=args.image_key
    )

if __name__ == "__main__":
    main()
