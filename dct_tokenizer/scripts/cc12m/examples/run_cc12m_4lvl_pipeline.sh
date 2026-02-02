#!/usr/bin/env bash
set -euo pipefail

TOK_DIR=/root/cc12m_tokens_4lvl/token_shards
OUT_TOK=/root/cc12m_tokens_4lvl
IDX_ROOT=/root/cc12m_indices_4lvl
LOG=/root/cc12m_4lvl_pipeline.log

exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] Starting 4-level tokenization (this will be large)..."

# Tokenize 4-level
rm -rf "$OUT_TOK"
python3 /root/cc12m_tokenize_pyramid_residual_4lvl.py \
  --shards-dir /root/cc12m_wds \
  --out-dir "$OUT_TOK" \
  --workers 32 \
  --concat

echo "[$(date)] Tokenization complete. Building shard indices..."
rm -rf "$IDX_ROOT"
python3 /root/build_cc12m_shard_indices.py \
  --token-shards "$TOK_DIR" \
  --out-root "$IDX_ROOT" \
  --workers 8 \
  --tokens-per-image 16320

echo "[$(date)] Index build complete. Generating samples (may be slow)..."
python3 /root/cc12m_generate_4lvl.py \
  --index-root "$IDX_ROOT" \
  --config /root/cc12m_tokens_4lvl/config.json \
  --out-dir /root/cc12m_generated_4lvl \
  --num-images 2 \
  --temperature 0.9 \
  --top-k 50 \
  --max-context 64

echo "[$(date)] Done."