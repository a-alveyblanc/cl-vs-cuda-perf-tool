#!/bin/zsh

# Select the OpenCL platform/device used by pyopencl.create_some_context().
# Adjust this after checking `clinfo` or the PyOpenCL prompt once.
export PYOPENCL_CTX="${PYOPENCL_CTX:-0:1}"

# Optional but useful when the target OpenCL path is POCL's CUDA backend.
export POCL_CUDA_GPU_ARCH="${POCL_CUDA_GPU_ARCH:-sm_90}"

# Keep benchmark/artifact runs reproducible.
export LOOPY_NO_CACHE=1
export PYOPENCL_NO_CACHE=1
export CUDA_CACHE_DISABLE=1

python matmul_suite.py \
  --variant register \
  --shape 8192x8192x8192 \
  --bm 128 --bn 128 --bk 16 \
  --tm 8 --tn 8 \
  --use-cuda \
  --dump-artifacts \
  --artifact-dir artifacts \
  --cuda-arch sm_90 \
  --analyze \
  --plot-metric gflops,ptxas_registers_max \
  --plot-x backend
