#!/bin/bash
# Canonical build. --default-stream per-thread is REQUIRED for CUDA-graph decode capture.
cd ~/gemma-cuda-server
nvcc -O2 -arch=sm_110a --default-stream per-thread -I include \
  src/forward.cu src/draft.cu kernels/fp4_gemm.cu kernels/nvfp4_quant.cu kernels/elementwise.cu kernels/attention.cu \
  -lcublasLt -o build/forward "$@"
echo "build exit=$?"
