# CC12M (WebDataset) pipeline

This folder contains the **training-free pyramid-residual tokenizers**, shardwise index builders, and generators used for the CC12M WebDataset experiments.

We use CC12M **WebDataset shards** (10,000 images per `.tar`) because it downloads very fast as large objects (vs per-image URL scraping).

## Dataset layout (server)

Downloaded shards:

- `/root/cc12m_wds/00000.tar ... 00099.tar`  (100 shards = ~1,000,000 images)

Each shard contains files like:

- `00000000.jpg`, `00000000.json`, `00000000.txt`, ...

## Token formats

All token streams are:

- `uint16` tokens
- tokens in `[0..255]`
- separator token = `65535` after each image

### Pyramid residual (N-level)

We represent an image as a **coarse base image** plus **residual corrections** at increasing resolutions.

Residual values are stored as:

- `residual_q = clip(residual + 128, 0, 255)`

so that *no-change* is near `128`.

#### 2-level (8→16)

- base 8×8 RGB: 192 tokens
- residual 16×16 RGB: 768 tokens
- total: **960 tokens/image**

Script: `cc12m_tokenize_pyramid_residual.py`

#### 3-level (8→16→32)

- total: **4032 tokens/image**

Script: `cc12m_tokenize_pyramid_residual_3lvl.py`

#### 4-level (8→16→32→64)

- total: **16320 tokens/image**

Script: `cc12m_tokenize_pyramid_residual_4lvl.py`

#### 5-level (8→16→32→64→128)

A high-detail variant that models up to 128×128 (then upsample to 256 for display).

- total: **65472 tokens/image**

Script: `cc12m_tokenize_pyramid_residual_5lvl.py`

#### 6-level (4→8→16→32→64→128)

A practical 6-level variant that ends at 128×128 to keep sizes manageable.

- total: **65520 tokens/image**

Script: `cc12m_tokenize_pyramid_residual_6lvl.py`

> Note: A “true” 6-level starting from base=8 would end at 256×256 and would be extremely large (262,080 tokens/image).

## Shardwise indexing

We build **one Fastgram-compatible suffix-array index per shard**.

Script:

- `build_cc12m_shard_indices.py`

It produces, per shard directory:

- `table.0` (suffix array table)
- `offset.0` (document offsets)
- `tokenized.0` (symlink to the shard token file)

Example output:

- `/root/cc12m_indices_4lvl/00000/{table.0,offset.0,tokenized.0}`

## Fan-out + merge across shards

Fastgram supports multiple index dirs, but for many shards we use a fan-out wrapper:

- `multishard_engine.py`

It:

- loads shard dirs in groups (e.g. 10 shards/engine)
- fans out queries across groups
- merges counts correctly using Infini-gram `suffix_len` semantics

## Generation

Generation scripts sample a token sequence left-to-right using `ntd()` and then decode to an image.

- 2-level: `cc12m_generate.py`
- 3-level: `cc12m_generate_3lvl.py`
- 4-level: `cc12m_generate_4lvl.py`
- 6-level: `cc12m_generate_6lvl.py`

## Memorization check (exact match)

- `check_memorization_4lvl.py`

Re-tokenizes an image and checks if the exact token sequence exists in the corpus.

## Example pipeline scripts

See `examples/`:

- `run_cc12m_3lvl_pipeline.sh`
- `run_cc12m_4lvl_pipeline.sh`

These were used on the vast.ai instances and may need path tweaks on other machines.
