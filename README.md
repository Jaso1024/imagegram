# Imagegram

N-gram based image generation using [FlexTok](https://github.com/apple/ml-flextok) tokenization and [Fastgram](https://github.com/Jaso1024/Fastgram) indexing.

## Overview

This project explores whether n-gram statistics over large image corpora can be used for image generation:

1. **Tokenize images** using FlexTok (converts images to 256 ordered tokens)
2. **Build n-gram index** using Fastgram (suffix array based)
3. **Generate images** by sampling from next-token distributions
4. **Decode tokens** back to images using FlexTok's decoder

## Installation

```bash
# Install dependencies
pip install fast-gram torch torchvision transformers accelerate datasets

# Clone and install FlexTok
git clone https://github.com/apple/ml-flextok
cd ml-flextok && pip install -e .
cd ..

# Build Fastgram index tools
git clone https://github.com/Jaso1024/Fastgram
cd Fastgram && mkdir build && cd build && cmake .. && make -j$(nproc)
cd ../..
```

## Usage

### 1. Tokenize a dataset

```bash
python tokenize_fast.py
```

### 2. Build the n-gram index

```bash
# Create input directory
mkdir -p index_input
python -c "
import numpy as np
tokens = np.load('tokens_flat.npy')
tokens.astype(np.uint16).tofile('index_input/tokenized.0')
"

# Build index
./Fastgram/build/tg_build_index index_input/ index/ 2 4 full
```

### 3. Generate images

```bash
python generate_images.py
```

## Key Findings

With 225k images (food101 + tiny-imagenet + cifar100):
- **54% of generated images are exact copies** of training images
- **46% are novel but often incoherent**
- N-gram model either memorizes or produces chaos

Hypothesis: More data (1M+ images) should improve n-gram coverage and reduce memorization.

## Scripts

| Script | Description |
|--------|-------------|
| `tokenize_fast.py` | Tokenize images with FlexTok using DataLoader |
| `overnight_run.py` | Full pipeline: tokenize → index → generate |
| `generate_images.py` | Generate images from n-gram sampling |
| `tokenize_imagenet_large.py` | Tokenize ImageNet 1.28M images |

## Architecture

```
[Images] → [FlexTok Encoder] → [Token Sequences] → [Fastgram Index]
                                                          ↓
[FlexTok Decoder] ← [Sampled Tokens] ← [ntd() sampling] ←─┘
```

## References

- [FlexTok Paper](https://arxiv.org/abs/2502.13967) - Variable-length image tokenization
- [Fastgram](https://github.com/Jaso1024/Fastgram) - Fast n-gram engine
- [InfiniGram](https://infini-gram.io/) - Inspiration for n-gram approach
