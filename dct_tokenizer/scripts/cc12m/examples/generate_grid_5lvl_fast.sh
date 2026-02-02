#!/usr/bin/env bash
set -euo pipefail

PY=/root/imagegram/dct_tokenizer/scripts/cc12m/cc12m_generate_5lvl.py
IDX=/root/cc12m_indices_5lvl
CFG=/root/cc12m_tokens_5lvl/config.json
BASE_OUT=/workspace/cc12m/gen/5lvl_grid_fast

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

COMMON=(--group-size 20 --threads-per-engine 1)

# 1 image per setting to keep runtime manageable
run_one t065_k5_ctx64    --num-images 1 --temperature 0.65 --top-k 5  --max-context 64  "${COMMON[@]}"
run_one t075_k10_ctx64   --num-images 1 --temperature 0.75 --top-k 10 --max-context 64  "${COMMON[@]}"
run_one t085_k20_ctx32   --num-images 1 --temperature 0.85 --top-k 20 --max-context 32  "${COMMON[@]}"
run_one t095_k40_ctx32   --num-images 1 --temperature 0.95 --top-k 40 --max-context 32  "${COMMON[@]}"
run_one t105_k80_ctx32   --num-images 1 --temperature 1.05 --top-k 80 --max-context 32  "${COMMON[@]}"

echo "All done. Outputs in $BASE_OUT"