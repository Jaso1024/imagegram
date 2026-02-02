#!/usr/bin/env bash
set -euo pipefail

# Example pipeline for 5-level (8->128) CC12M pyramid-residual.
# Assumes:
#   - CC12M shards exist in /root/cc12m_wds (00000.tar..00099.tar)
#   - fast_build_index compiled at /root/fast_build_index
#   - Fastgram python package installed (for generation)

SHARDS_DIR=/root/cc12m_wds
TOK_DIR=/root/cc12m_tokens_5lvl
IDX_DIR=/root/cc12m_indices_5lvl
LOG=/root/cc12m_5lvl_pipeline.log

exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] Starting 5-level tokenization..."
rm -rf "$TOK_DIR"
python3 /root/imagegram/dct_tokenizer/scripts/cc12m/cc12m_tokenize_pyramid_residual_5lvl.py \
  --shards-dir "$SHARDS_DIR" \
  --out-dir "$TOK_DIR" \
  --workers 32

echo "[$(date)] Tokenization complete. Deleting original tar shards to save space..."
rm -rf "$SHARDS_DIR"

echo "[$(date)] Building shardwise indices..."
rm -rf "$IDX_DIR"
python3 /root/imagegram/dct_tokenizer/scripts/cc12m/build_cc12m_shard_indices.py \
  --token-shards "$TOK_DIR/token_shards" \
  --out-root "$IDX_DIR" \
  --workers 8 \
  --tokens-per-image 65472 \
  --token-width 2 \
  --fast-build-index /root/fast_build_index

echo "[$(date)] Generating a few samples (slow)..."
python3 /root/imagegram/dct_tokenizer/scripts/cc12m/cc12m_generate_5lvl.py \
  --index-root "$IDX_DIR" \
  --config "$TOK_DIR/config.json" \
  --out-dir /root/cc12m_generated_5lvl \
  --num-images 1 \
  --temperature 0.9 \
  --top-k 20 \
  --max-context 32

echo "[$(date)] Done."