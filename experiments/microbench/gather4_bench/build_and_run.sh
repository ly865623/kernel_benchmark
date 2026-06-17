#!/bin/bash
# Build + run the faithful FlashMLA KV-gather (TMA tile::gather4) microbench on B200.
set -e
FM=/home/liuy/code/FlashMLA
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/gather4_tput.out"

nvcc -O3 -std=c++17 \
  -gencode arch=compute_100f,code=sm_100f \
  --expt-relaxed-constexpr --expt-extended-lambda \
  -I"$FM/csrc" \
  -I"$FM/csrc/kerutils/include" \
  -I"$FM/csrc/cutlass/include" \
  -I"$FM/csrc/cutlass/tools/util/include" \
  "$HERE/gather4_tput.cu" -o "$OUT" -lcuda

echo "BUILT: $OUT"
"$OUT" "$@"
