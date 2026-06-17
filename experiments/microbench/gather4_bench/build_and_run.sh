#!/bin/bash
# Build + run the faithful FlashMLA KV-gather (TMA tile::gather4) microbench on B200.
# FlashMLA headers come from the in-repo submodule thirdparty/FlashMLA by default
# (populate with: git submodule update --init --recursive); override with FM=/path/to/FlashMLA.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
FM="${FM:-$(cd "$HERE/../../.." && pwd)/thirdparty/FlashMLA}"
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
