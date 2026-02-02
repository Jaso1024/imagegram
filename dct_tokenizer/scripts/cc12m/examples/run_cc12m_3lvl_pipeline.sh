#!/usr/bin/env bash
set -euo pipefail

TOK_DIR=/root/cc12m_tokens_3lvl/token_shards
IDX_ROOT=/root/cc12m_indices_3lvl
LOG=/root/cc12m_3lvl_pipeline.log

exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] Waiting for 3-level tokenization to finish..."

while true; do
  bins=$(ls "$TOK_DIR" 2>/dev/null | grep -c '\.bin$' || true)
  tmps=$(ls "$TOK_DIR" 2>/dev/null | grep -c '\.tmp$' || true)
  echo "[$(date)] bins=$bins tmps=$tmps"
  if [[ "$bins" -ge 100 && "$tmps" -eq 0 ]]; then
    break
  fi
  sleep 30
done

echo "[$(date)] Tokenization complete. Building shard indices..."
rm -rf "$IDX_ROOT"
python3 /root/build_cc12m_shard_indices.py \
  --token-shards "$TOK_DIR" \
  --out-root "$IDX_ROOT" \
  --workers 16 \
  --tokens-per-image 4032

echo "[$(date)] Index build complete. Generating samples..."
python3 /root/cc12m_generate_3lvl.py \
  --index-root "$IDX_ROOT" \
  --config /root/cc12m_tokens_3lvl/config.json \
  --out-dir /root/cc12m_generated_3lvl \
  --num-images 8 \
  --temperature 0.9 \
  --top-k 50 \
  --max-context 64

echo "[$(date)] Done."