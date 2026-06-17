#!/bin/bash
# Build + run the faithful FlashMLA sm100 CORRECTION-stage FP32-ALU microbench on B200.
# Pure CUDA, register-resident, no cutlass/FlashMLA headers needed.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/corr_rescale_tput.out"

nvcc -O3 -std=c++17 \
  -gencode arch=compute_100f,code=sm_100f \
  "$HERE/corr_rescale_tput.cu" -o "$OUT" -lcuda

echo "BUILT: $OUT"
"$OUT" "$@"
