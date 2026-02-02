#!/usr/bin/env bash
set -euo pipefail

PY=/root/imagegram/dct_tokenizer/scripts/cc12m/cc12m_generate_5lvl.py
IDX=/root/cc12m_indices_5lvl
CFG=/root/cc12m_tokens_5lvl/config.json
BASE_OUT=/workspace/cc12m/gen/5lvl_grid

mkdir -p "$BASE_OUT"

run_one() {
  name="$1"; shift
  out="$BASE_OUT/$name"
  mkdir -p "$out"

  {
    echo "=== $name ==="
    echo "python3 $PY --index-root $IDX --config $CFG --out-dir $out $*"
  } | tee "$out/params.txt"

  python3 "$PY" --index-root "$IDX" --config "$CFG" --out-dir "$out" "$@"
}

# Use fewer engines by grouping 20 shards/engine (100 shards => 5 engines)
COMMON=(--group-size 20 --threads-per-engine 1)

# 2 images each
run_one t065_k5_ctx128  --num-images 2 --temperature 0.65 --top-k 5  --max-context 128 "${COMMON[@]}"
run_one t075_k10_ctx128 --num-images 2 --temperature 0.75 --top-k 10 --max-context 128 "${COMMON[@]}"
run_one t085_k20_ctx64  --num-images 2 --temperature 0.85 --top-k 20 --max-context 64  "${COMMON[@]}"
run_one t095_k40_ctx64  --num-images 2 --temperature 0.95 --top-k 40 --max-context 64  "${COMMON[@]}"

echo "All done. Outputs in $BASE_OUT"