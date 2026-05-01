# Important: do not leave this set to 0.
unset POCL_CACHE_DIR
unset POCL_KERNEL_CACHE
unset RUN

export PYOPENCL_CTX=0:1
export LOOPY_NO_CACHE=1
export PYOPENCL_NO_CACHE=1
export CUDA_CACHE_DISABLE=1

python matmul_dump_artifacts.py \
  --m 8192 --n 8192 --k 8192 \
  --bm 128 --bn 128 --bk 16 \
  --tm 8 --tn 8 \
  --register-tiled \
  --use-cuda \
  --dump-artifacts \
  --artifact-dir artifacts \
  --cuda-arch sm_90

export RUN="$(ls -td artifacts/* | head -1)"
python analyze_compiler_artifacts.py "$RUN" --print-files --color always | less -R
