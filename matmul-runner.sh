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

#!/usr/bin/env bash
# Simple matrix-size sweep for matmul_suite.py.
#
# This is the normal runner. It uses matmul_suite.py directly, the same way the
# finite-difference driver handles size sweeps.
#
# Examples:
#   ./run_matmul_sizes_simple.sh
#   SIZES=1024,2048,4096 BACKENDS=cuda ./run_matmul_sizes_simple.sh
#   SHAPES="1024x1024x1024 2048x4096x1024" BACKENDS=opencl,cuda ./run_matmul_sizes_simple.sh

SIZES="${SIZES:-1024,2048,4096,8192}"
SHAPES="${SHAPES:-}"
BACKENDS="${BACKENDS:-opencl,cuda}"

VARIANT="${VARIANT:-register}"
BM="${BM:-128}"
BN="${BN:-128}"
BK="${BK:-16}"
TM="${TM:-8}"
TN="${TN:-8}"

CUDA_ARCH="${CUDA_ARCH:-sm_90}"
ARTIFACT_DIR="${ARTIFACT_DIR:-compiler_artifacts}"
SUITE_NAME="${SUITE_NAME:-matmul_sizes}"

NWARMUP="${NWARMUP:-5}"
NITERATIONS="${NITERATIONS:-100}"

MATMUL_SUITE="${MATMUL_SUITE:-./matmul_suite.py}"

shape_args=()
if [[ -n "${SHAPES}" ]]; then
  for shape in ${SHAPES}; do
    shape_args+=(--shape "${shape}")
  done
else
  IFS=',' read -r -a size_array <<< "${SIZES}"
  for size in "${size_array[@]}"; do
    size="${size//[[:space:]]/}"
    [[ -z "${size}" ]] && continue
    shape_args+=(--shape "${size}x${size}x${size}")
  done
fi

python "${MATMUL_SUITE}" \
  "${shape_args[@]}" \
  --variant "${VARIANT}" \
  --bm "${BM}" --bn "${BN}" --bk "${BK}" \
  --tm "${TM}" --tn "${TN}" \
  --backend "${BACKENDS}" \
  --cuda-arch "${CUDA_ARCH}" \
  --artifact-dir "${ARTIFACT_DIR}" \
  --suite-name "${SUITE_NAME}" \
  --nwarmup "${NWARMUP}" \
  --niterations "${NITERATIONS}" \
  --dump-artifacts \
  --analyze \
  --plot-metric gflops \
  --plot-metric ptxas_registers_max \
  --plot-x metadata.m \
  --plot-series backend
