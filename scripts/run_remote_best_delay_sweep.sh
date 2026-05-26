#!/usr/bin/env bash
set -euo pipefail

RUNNER=/data/danyang/mooncake-contention/scripts/run_mooncake_case.py

run_case() {
  local label=$1
  shift
  echo "===== best-delay ${label} ====="
  "$RUNNER" "$@" --bucket "bestdelay_${label}"
}

run_delay_grid() {
  local label=$1
  local prompt=$2
  local conc=$3
  local max_len=$4
  local max_batch=$5
  local blocks=$6
  local tokens=${7:-32}
  shift 7
  local common=(
    --case storepd
    --prompt-tokens "$prompt"
    --concurrency "$conc"
    --workers 8
    --launch-gap-s 0.05
    --stream-decode
    --decode-max-tokens "$tokens"
    --decode-start-delay-s 0.2
    --max-model-len "$max_len"
    --max-num-batched-tokens "$max_batch"
    --num-gpu-blocks "$blocks"
  )
  if [[ "$conc" -gt 1 ]]; then
    common+=(--unique-prompts)
  fi
  run_case "${label}_off" "${common[@]}" --protect-mode off
  for delay in "$@"; do
    run_case "${label}_d${delay}" "${common[@]}" \
      --protect-mode delay_store --protect-grace-ms 10 --protect-max-wait-ms "$delay"
  done
}

# A compact sweep that spans KV size and serving shape.
run_delay_grid 8k_c1 8000 1 8192 8192 1024 64 50 100 200 500
run_delay_grid 16k_c1 16000 1 16384 16384 4096 64 50 100 200 500
run_delay_grid 16k_c2 16000 2 16384 32768 4096 32 50 100 200 500
run_delay_grid 24k_c2 24000 2 24576 49152 8192 32 50 100 200 500
run_delay_grid 16k_c3 16000 3 16384 49152 8192 32 50 100 200 500
run_delay_grid 8k_c8 8000 8 8192 65536 8192 32 100 200 500
