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

python finite_difference_2_5d_suite.py \
  --variant all \
  --npoints 64,128,256 \
  --stencil-width 5,9 \
  --use-cuda \
  --validate \
  --analyze \
  --plot-metric gflops,ptxas_registers_max \
  --plot-x metadata.npts \
  --plot-series metadata.variant
