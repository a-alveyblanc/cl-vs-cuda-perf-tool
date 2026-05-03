#!/bin/bash

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
  --backend opencl \
  --variant all \
  --npoints 32,64,128,256,512 \
  --stencil-width 3,5,9 \
  --analyze

python finite_difference_2_5d_suite.py \
  --backend cuda \
  --variant all \
  --npoints 32,64,128,256,512 \
  --stencil-width 3,5,9 \
  --validate \
  --analyze
