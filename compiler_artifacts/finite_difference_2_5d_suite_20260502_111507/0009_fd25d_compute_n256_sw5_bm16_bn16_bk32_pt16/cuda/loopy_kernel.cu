#define bIdx(N) ((int) blockIdx.N)
#define tIdx(N) ((int) threadIdx.N)
#define LOOPY_CALL_WITH_INTEGER_TYPES(MACRO_NAME) \
    MACRO_NAME(int8, char) \
    MACRO_NAME(int16, short) \
    MACRO_NAME(int32, int) \
    MACRO_NAME(int64, long)
#define LOOPY_DEFINE_FLOOR_DIV_POS_B(SUFFIX, TYPE) \
    inline __device__ TYPE loopy_floor_div_pos_b_##SUFFIX(TYPE a, TYPE b) \
    { \
        if (a<0) \
            a = a - (b-1); \
        return a/b; \
    }
LOOPY_CALL_WITH_INTEGER_TYPES(LOOPY_DEFINE_FLOOR_DIV_POS_B)
#undef LOOPY_DEFINE_FLOOR_DIV_POS_B
#undef LOOPY_CALL_WITH_INTEGER_TYPES

extern "C" __global__ void __launch_bounds__(256) loopy_kernel(double const *__restrict__ u, double *__restrict__ lap_u, double const *__restrict__ c)
{
  double acc_l;
  __shared__ double u_ij_plane[20 * 20];
  double u_k_buf[5];

  for (int ki = ((-1 + bIdx(x) >= 0) ? 0 : 2); ki <= ((-7 + bIdx(x) == 0) ? 29 : 31); ++ki)
  {
    if (-2 + tIdx(x) + 16 * bIdx(y) >= 0 && 253 + -1 * tIdx(x) + -16 * bIdx(y) >= 0 && -2 + tIdx(y) + 16 * bIdx(z) >= 0 && 253 + -1 * tIdx(y) + -16 * bIdx(z) >= 0)
      acc_l = (double) (0.0);
    __syncthreads() /* for u_ij_plane (u_plane_compute rev-depends on insn_l_update) */;
    for (int ii_s_tile = (((-26 + 13 * tIdx(y) + 240 * bIdx(z) >= 0 && 1 + -1 * tIdx(y) + -1 * tIdx(x) >= 0 && -1 + bIdx(y) >= 0 && 239 + tIdx(y) + -1 * tIdx(x) + -16 * bIdx(y) >= 0) || (-26 + 13 * tIdx(y) + 240 * bIdx(z) >= 0 && -2 + tIdx(y) + tIdx(x) >= 0 && -4 + tIdx(y) + 16 * bIdx(z) + tIdx(x) >= 0 && -4 + tIdx(y) + tIdx(x) + 16 * bIdx(y) >= 0 && -6 + tIdx(y) + 16 * bIdx(z) + tIdx(x) + 16 * bIdx(y) >= 0 && -26 + 13 * tIdx(x) + 240 * bIdx(y) >= 0) || (bIdx(y) == 0 && -26 + 13 * tIdx(y) + 240 * bIdx(z) >= 0 && 1 + -1 * tIdx(y) + -1 * tIdx(x) >= 0) || (bIdx(y) == 0 && 253 + -1 * tIdx(y) + -16 * bIdx(z) + tIdx(x) >= 0 && -4 + tIdx(y) + tIdx(x) >= 0 && -6 + tIdx(y) + 16 * bIdx(z) + tIdx(x) >= 0 && 1 + -1 * tIdx(x) >= 0) || (bIdx(z) == 0 && -2 + tIdx(y) >= 0 && 3 + -1 * tIdx(y) + -1 * tIdx(x) >= 0 && bIdx(y) >= 0 && 237 + tIdx(y) + -1 * tIdx(x) + -16 * bIdx(y) >= 0) || (bIdx(y) == 0 && -1 + bIdx(z) >= 0 && tIdx(x) >= 0 && -2 + tIdx(y) + tIdx(x) >= 0 && 3 + -1 * tIdx(y) + -1 * tIdx(x) >= 0 && 1 + tIdx(y) + -1 * tIdx(x) >= 0) || (bIdx(y) == 0 && -15 + bIdx(z) == 0 && -14 + tIdx(y) + -1 * tIdx(x) >= 0) || (bIdx(y) == 0 && bIdx(z) == 0 && tIdx(x) >= 0 && -4 + tIdx(y) + tIdx(x) >= 0 && 5 + -1 * tIdx(y) + -1 * tIdx(x) >= 0 && -1 + tIdx(y) + -1 * tIdx(x) >= 0)) ? 0 : 1); ii_s_tile <= (((-15 + bIdx(z) == 0 && -4 + tIdx(y) >= 0 && 14 + -1 * tIdx(y) >= 0 && -1 + -1 * bIdx(y) >= 0 && 25 + -13 * tIdx(x) + -240 * bIdx(y) >= 0) || (-15 + bIdx(z) == 0 && -4 + tIdx(y) >= 0 && 14 + -1 * tIdx(y) >= 0 && 14 + -1 * bIdx(y) >= 0 && -26 + 13 * tIdx(x) + 240 * bIdx(y) >= 0) || (-15 + bIdx(z) == 0 && -2 + tIdx(y) >= 0 && 3 + -1 * tIdx(y) >= 0 && -1 + -1 * bIdx(y) >= 0 && 25 + -13 * tIdx(x) + -240 * bIdx(y) >= 0) || (-15 + bIdx(z) == 0 && -2 + tIdx(y) >= 0 && 3 + -1 * tIdx(y) >= 0 && 14 + -1 * bIdx(y) >= 0 && -26 + 13 * tIdx(x) + 240 * bIdx(y) >= 0) || (-15 + bIdx(z) == 0 && -4 + tIdx(y) >= 0 && 14 + -1 * tIdx(y) >= 0 && bIdx(y) >= 0 && 14 + -1 * bIdx(y) >= 0 && 25 + -13 * tIdx(x) + -240 * bIdx(y) >= 0) || (-15 + bIdx(z) == 0 && -2 + tIdx(y) >= 0 && 3 + -1 * tIdx(y) >= 0 && bIdx(y) >= 0 && 14 + -1 * bIdx(y) >= 0 && 25 + -13 * tIdx(x) + -240 * bIdx(y) >= 0) || (-15 + bIdx(y) == 0 && -15 + bIdx(z) == 0 && -4 + tIdx(y) >= 0 && 14 + -1 * tIdx(y) >= 0) || (-15 + bIdx(y) == 0 && -15 + bIdx(z) == 0 && -2 + tIdx(y) >= 0 && 3 + -1 * tIdx(y) >= 0) || (-15 + bIdx(y) == 0 && -3 + tIdx(y) == 0 && -1 + -1 * bIdx(z) >= 0 && 1 + -1 * tIdx(x) >= 0 && -240 * bIdx(z) + 13 * tIdx(x) >= 0) || (bIdx(y) == 0 && -3 + tIdx(y) == 0 && -1 + -1 * bIdx(z) >= 0 && -2 + tIdx(x) >= 0 && 3 + -1 * tIdx(x) >= 0) || (bIdx(y) == 0 && -3 + tIdx(y) == 0 && 14 + -1 * bIdx(z) >= 0 && -2 + tIdx(x) >= 0 && 3 + -1 * tIdx(x) >= 0 && -1 + 240 * bIdx(z) + -13 * tIdx(x) >= 0) || (-15 + bIdx(y) == 0 && -3 + tIdx(y) == 0 && -1 + -1 * bIdx(z) >= 0 && -1 + 240 * bIdx(z) + -13 * tIdx(x) >= 0 && 16 + tIdx(x) + 16 * loopy_floor_div_pos_b_int32(-1 + -1 * tIdx(x), 16) >= 0 && -15 + -1 * tIdx(x) + -16 * loopy_floor_div_pos_b_int32(-1 + -1 * tIdx(x), 16) >= 0) || (-15 + bIdx(y) == 0 && -3 + tIdx(y) == 0 && bIdx(z) >= 0 && 14 + -1 * bIdx(z) >= 0 && 1 + -1 * tIdx(x) >= 0 && -1 + 240 * bIdx(z) + -13 * tIdx(x) >= 0 && 16 + tIdx(x) + 16 * loopy_floor_div_pos_b_int32(-1 + -1 * tIdx(x), 16) >= 0 && -15 + -1 * tIdx(x) + -16 * loopy_floor_div_pos_b_int32(-1 + -1 * tIdx(x), 16) >= 0) || (-15 + bIdx(y) == 0 && bIdx(z) == 0 && -3 + tIdx(y) == 0 && tIdx(x) >= 0 && 1 + -1 * tIdx(x) >= 0) || (bIdx(y) == 0 && -15 + bIdx(z) == 0 && -1 + tIdx(y) == 0 && -2 + tIdx(x) >= 0 && 3 + -1 * tIdx(x) >= 0) || (bIdx(y) == 0 && bIdx(z) == 0 && -3 + tIdx(y) == 0 && -2 + tIdx(x) >= 0 && 3 + -1 * tIdx(x) >= 0) || (-15 + bIdx(y) == 0 && -15 + bIdx(z) == 0 && -1 + tIdx(y) == 0 && 1 + -1 * tIdx(x) >= 0 && 16 + tIdx(x) + 16 * loopy_floor_div_pos_b_int32(-1 + -1 * tIdx(x), 16) >= 0 && -15 + -1 * tIdx(x) + -16 * loopy_floor_div_pos_b_int32(-1 + -1 * tIdx(x), 16) >= 0)) ? 0 : (((-4 + tIdx(y) >= 0 && 14 + -1 * bIdx(z) >= 0 && -26 + 13 * tIdx(y) + 240 * bIdx(z) >= 0 && 14 + -1 * bIdx(y) >= 0 && -26 + 13 * tIdx(x) + 240 * bIdx(y) >= 0 && 12 + tIdx(y) + 16 * loopy_floor_div_pos_b_int32(3 + -1 * tIdx(y), 16) >= 0 && -1 * tIdx(y) + -16 * loopy_floor_div_pos_b_int32(3 + -1 * tIdx(y), 16) >= 0) || (3 + -1 * tIdx(y) >= 0 && 14 + -1 * bIdx(z) >= 0 && -26 + 13 * tIdx(y) + 240 * bIdx(z) >= 0 && 14 + -1 * bIdx(y) >= 0 && -26 + 13 * tIdx(x) + 240 * bIdx(y) >= 0 && 12 + tIdx(y) + 16 * loopy_floor_div_pos_b_int32(3 + -1 * tIdx(y), 16) >= 0 && -1 * tIdx(y) + -16 * loopy_floor_div_pos_b_int32(3 + -1 * tIdx(y), 16) >= 0) || (-1 + tIdx(y) >= 0 && 3 + -1 * tIdx(y) >= 0 && 14 + -1 * bIdx(z) >= 0 && -1 + 240 * bIdx(z) + -13 * tIdx(x) >= 0 && -1 + bIdx(y) >= 0 && 15 + -1 * bIdx(y) >= 0 && 47 + -1 * tIdx(y) + tIdx(x) + -2 * bIdx(y) + 16 * loopy_floor_div_pos_b_int32(2 + -1 * tIdx(y) + -1 * tIdx(x), 16) >= 0 && 13 + tIdx(y) + tIdx(x) + 16 * loopy_floor_div_pos_b_int32(2 + -1 * tIdx(y) + -1 * tIdx(x), 16) >= 0 && 2 + -1 * tIdx(y) + -1 * tIdx(x) + -16 * loopy_floor_div_pos_b_int32(2 + -1 * tIdx(y) + -1 * tIdx(x), 16) >= 0) || (bIdx(y) == 0 && -1 + tIdx(y) >= 0 && 3 + -1 * tIdx(y) >= 0 && 14 + -1 * bIdx(z) >= 0 && -11 + tIdx(x) >= 0 && -1 + 240 * bIdx(z) + -13 * tIdx(x) >= 0) || (bIdx(z) == 0 && -1 + tIdx(y) >= 0 && 3 + -1 * tIdx(y) >= 0 && -2 + tIdx(x) >= 0 && 15 + -1 * bIdx(y) >= 0 && -4 + tIdx(x) + 16 * bIdx(y) >= 0) || (-15 + bIdx(y) == 0 && -4 + tIdx(y) >= 0 && 14 + -1 * bIdx(z) >= 0 && -26 + 13 * tIdx(y) + 240 * bIdx(z) >= 0 && 12 + tIdx(y) + 16 * loopy_floor_div_pos_b_int32(3 + -1 * tIdx(y), 16) >= 0 && -1 * tIdx(y) + -16 * loopy_floor_div_pos_b_int32(3 + -1 * tIdx(y), 16) >= 0) || (-15 + bIdx(y) == 0 && 3 + -1 * tIdx(y) >= 0 && 14 + -1 * bIdx(z) >= 0 && -26 + 13 * tIdx(y) + 240 * bIdx(z) >= 0 && 12 + tIdx(y) + 16 * loopy_floor_div_pos_b_int32(3 + -1 * tIdx(y), 16) >= 0 && -1 * tIdx(y) + -16 * loopy_floor_div_pos_b_int32(3 + -1 * tIdx(y), 16) >= 0) || (bIdx(z) == 0 && -1 + tIdx(y) >= 0 && 3 + -1 * tIdx(y) >= 0 && 1 + -1 * tIdx(x) >= 0 && bIdx(y) >= 0 && 17 + -1 * tIdx(y) + -1 * bIdx(y) >= 0 && 15 + -1 * bIdx(y) >= 0) || (bIdx(y) == 0 && -4 + tIdx(y) >= 0 && 14 + -1 * bIdx(z) >= 0 && -26 + 13 * tIdx(y) + 240 * bIdx(z) >= 0 && 1 + -1 * tIdx(x) >= 0 && 12 + tIdx(y) + 16 * loopy_floor_div_pos_b_int32(3 + -1 * tIdx(y), 16) >= 0 && -1 * tIdx(y) + -16 * loopy_floor_div_pos_b_int32(3 + -1 * tIdx(y), 16) >= 0) || (bIdx(y) == 0 && 3 + -1 * tIdx(y) >= 0 && 14 + -1 * bIdx(z) >= 0 && -26 + 13 * tIdx(y) + 240 * bIdx(z) >= 0 && 1 + -1 * tIdx(x) >= 0 && 12 + tIdx(y) + 16 * loopy_floor_div_pos_b_int32(3 + -1 * tIdx(y), 16) >= 0 && -1 * tIdx(y) + -16 * loopy_floor_div_pos_b_int32(3 + -1 * tIdx(y), 16) >= 0) || (bIdx(y) == 0 && -1 + tIdx(y) >= 0 && 3 + -1 * tIdx(y) >= 0 && 14 + -1 * bIdx(z) >= 0 && 10 + -1 * tIdx(x) >= 0 && -1 + 240 * bIdx(z) + -13 * tIdx(x) >= 0 && 15 + -1 * tIdx(y) + tIdx(x) + 16 * loopy_floor_div_pos_b_int32(4 + -1 * tIdx(y) + -1 * tIdx(x), 16) >= 0 && 11 + tIdx(y) + tIdx(x) + 16 * loopy_floor_div_pos_b_int32(4 + -1 * tIdx(y) + -1 * tIdx(x), 16) >= 0 && 4 + -1 * tIdx(y) + -1 * tIdx(x) + -16 * loopy_floor_div_pos_b_int32(4 + -1 * tIdx(y) + -1 * tIdx(x), 16) >= 0) || (bIdx(z) == 0 && tIdx(y) == 0 && 14 + -1 * bIdx(y) >= 0 && -4 + tIdx(x) + 16 * bIdx(y) >= 0) || (bIdx(y) == 0 && bIdx(z) == 0 && -1 + tIdx(y) >= 0 && 2 + -1 * tIdx(y) >= 0 && -2 + tIdx(x) >= 0 && 3 + -1 * tIdx(x) >= 0) || (-15 + bIdx(y) == 0 && bIdx(z) == 0 && tIdx(y) == 0) || (bIdx(y) == 0 && bIdx(z) == 0 && tIdx(y) == 0 && 3 + -1 * tIdx(x) >= 0)) ? 1 + -1 * tIdx(y) + (3 + 15 * tIdx(y)) / 16 : -12 + -1 * tIdx(y) + (209 + 15 * tIdx(y)) / 16)); ++ii_s_tile)
      for (int ji_s_tile = ((255 + -1 * tIdx(y) + -16 * ii_s_tile + -16 * bIdx(z) + tIdx(x) >= 0 && 17 + -1 * tIdx(y) + -16 * ii_s_tile + tIdx(x) >= 0 && -2 + tIdx(y) + 16 * ii_s_tile + tIdx(x) >= 0 && -4 + tIdx(y) + 16 * ii_s_tile + 16 * bIdx(z) + tIdx(x) >= 0 && 253 + -1 * tIdx(y) + -16 * ii_s_tile + -16 * bIdx(z) + tIdx(x) + 16 * bIdx(y) >= 0 && 15 + -1 * tIdx(y) + -16 * ii_s_tile + tIdx(x) + 16 * bIdx(y) >= 0 && -4 + tIdx(y) + 16 * ii_s_tile + tIdx(x) + 16 * bIdx(y) >= 0 && -6 + tIdx(y) + 16 * ii_s_tile + 16 * bIdx(z) + tIdx(x) + 16 * bIdx(y) >= 0 && -26 + 13 * tIdx(x) + 240 * bIdx(y) >= 0) ? 0 : 1); ji_s_tile <= ((-15 + bIdx(y) == 0 && -15 + bIdx(z) == 0 && -15 + tIdx(y) + 16 * ii_s_tile >= 0 && 17 + -1 * tIdx(y) + -16 * ii_s_tile >= 0 && 48 + -1 * tIdx(y) + -32 * ii_s_tile + -1 * tIdx(x) + 16 * loopy_floor_div_pos_b_int32(-1 + -1 * tIdx(y) + tIdx(x), 16) >= 0 && 16 + tIdx(y) + -1 * tIdx(x) + 16 * loopy_floor_div_pos_b_int32(-1 + -1 * tIdx(y) + tIdx(x), 16) >= 0 && -1 + -1 * tIdx(y) + tIdx(x) + -16 * loopy_floor_div_pos_b_int32(-1 + -1 * tIdx(y) + tIdx(x), 16) >= 0) ? 2 + -1 * tIdx(y) + -1 * ii_s_tile + -1 * tIdx(x) + (15 * tIdx(y) + 15 * tIdx(x)) / 16 : (((ii_s_tile == 0 && 1 + -1 * tIdx(y) >= 0 && -26 + 13 * tIdx(y) + 240 * bIdx(z) >= 0 && -1 + bIdx(y) >= 0 && 14 + -1 * bIdx(y) >= 0) || (bIdx(y) == 0 && ii_s_tile == 0 && 1 + -1 * tIdx(y) >= 0 && -26 + 13 * tIdx(y) + 240 * bIdx(z) >= 0 && 17 + tIdx(y) + -1 * tIdx(x) + 16 * loopy_floor_div_pos_b_int32(-4 + tIdx(y) + tIdx(x), 16) >= 0 && -4 + tIdx(y) + tIdx(x) + -16 * loopy_floor_div_pos_b_int32(-4 + tIdx(y) + tIdx(x), 16) >= 0)) ? 1 + -1 * tIdx(x) + (1 + tIdx(y) + 15 * tIdx(x)) / 16 : (((-15 + bIdx(y) == 0 && -1 + ii_s_tile == 0 && -1 + tIdx(y) >= 0 && 14 + -1 * bIdx(z) >= 0 && 18 + -1 * tIdx(y) + -1 * tIdx(x) + 16 * loopy_floor_div_pos_b_int32(1 + -1 * tIdx(y) + tIdx(x), 16) >= 0 && 14 + tIdx(y) + -1 * tIdx(x) + 16 * loopy_floor_div_pos_b_int32(1 + -1 * tIdx(y) + tIdx(x), 16) >= 0 && 1 + -1 * tIdx(y) + tIdx(x) + -16 * loopy_floor_div_pos_b_int32(1 + -1 * tIdx(y) + tIdx(x), 16) >= 0) || (-15 + bIdx(z) == 0 && -1 + ii_s_tile == 0 && 1 + -1 * tIdx(y) >= 0 && bIdx(y) >= 0 && 14 + -1 * bIdx(y) >= 0 && 18 + -1 * tIdx(y) + -1 * tIdx(x) + 2 * bIdx(y) + 16 * loopy_floor_div_pos_b_int32(-3 + -1 * tIdx(y) + tIdx(x), 16) >= 0 && 18 + tIdx(y) + -1 * tIdx(x) + 16 * loopy_floor_div_pos_b_int32(-3 + -1 * tIdx(y) + tIdx(x), 16) >= 0 && -3 + -1 * tIdx(y) + tIdx(x) + -16 * loopy_floor_div_pos_b_int32(-3 + -1 * tIdx(y) + tIdx(x), 16) >= 0)) ? 1 + -1 * tIdx(y) + -1 * tIdx(x) + (2 + 15 * tIdx(y) + 15 * tIdx(x)) / 16 : ((-2 + tIdx(y) + 16 * ii_s_tile >= 0 && 17 + -1 * tIdx(y) + -16 * ii_s_tile >= 0 && 15 + -1 * ii_s_tile + -1 * bIdx(z) >= 0 && -4 + tIdx(y) + 16 * ii_s_tile + 16 * bIdx(z) >= 0 && 14 + -1 * bIdx(y) >= 0) ? 1 + -1 * tIdx(x) + (3 + 15 * tIdx(x)) / 16 : (((-1 + ii_s_tile == 0 && -2 + tIdx(y) >= 0 && 14 + -1 * bIdx(z) >= 0 && -1 + bIdx(y) >= 0 && 14 + -1 * bIdx(y) >= 0) || (bIdx(y) == 0 && -1 + ii_s_tile == 0 && -2 + tIdx(y) >= 0 && 14 + -1 * bIdx(z) >= 0 && 20 + -1 * tIdx(y) + -1 * tIdx(x) + 16 * loopy_floor_div_pos_b_int32(-1 + -1 * tIdx(y) + tIdx(x), 16) >= 0 && -1 + -1 * tIdx(y) + tIdx(x) + -16 * loopy_floor_div_pos_b_int32(-1 + -1 * tIdx(y) + tIdx(x), 16) >= 0)) ? 1 + -1 * tIdx(y) + -1 * tIdx(x) + (4 + 15 * tIdx(y) + 15 * tIdx(x)) / 16 : ((-15 + bIdx(y) == 0 && bIdx(z) == 0 && ii_s_tile == 0 && -2 + tIdx(y) >= 0 && 3 + -1 * tIdx(y) >= 0 && 12 + tIdx(y) + tIdx(x) + 16 * loopy_floor_div_pos_b_int32(-3 + tIdx(y) + -1 * tIdx(x), 16) >= 0 && -3 + tIdx(y) + -1 * tIdx(x) + -16 * loopy_floor_div_pos_b_int32(-3 + tIdx(y) + -1 * tIdx(x), 16) >= 0) ? -1 * tIdx(x) + (13 + tIdx(y) + 15 * tIdx(x)) / 16 : (((-15 + bIdx(y) == 0 && ii_s_tile == 0 && 2 + -1 * tIdx(y) >= 0 && -1 + bIdx(z) >= 0 && 17 + -1 * tIdx(y) + -1 * tIdx(x) + 16 * loopy_floor_div_pos_b_int32(-2 + tIdx(y) + tIdx(x), 16) >= 0 && 15 + tIdx(y) + -1 * tIdx(x) + 16 * loopy_floor_div_pos_b_int32(-2 + tIdx(y) + tIdx(x), 16) >= 0 && -2 + tIdx(y) + tIdx(x) + -16 * loopy_floor_div_pos_b_int32(-2 + tIdx(y) + tIdx(x), 16) >= 0) || (bIdx(z) == 0 && ii_s_tile == 0 && -2 + tIdx(y) >= 0 && 3 + -1 * tIdx(y) >= 0 && bIdx(y) >= 0 && 14 + -1 * bIdx(y) >= 0 && 21 + -1 * tIdx(y) + -1 * tIdx(x) + 16 * loopy_floor_div_pos_b_int32(-6 + tIdx(y) + tIdx(x), 16) >= 0 && 15 + tIdx(y) + -1 * tIdx(x) + 2 * bIdx(y) + 16 * loopy_floor_div_pos_b_int32(-6 + tIdx(y) + tIdx(x), 16) >= 0 && -6 + tIdx(y) + tIdx(x) + -16 * loopy_floor_div_pos_b_int32(-6 + tIdx(y) + tIdx(x), 16) >= 0)) ? -1 * tIdx(x) + (15 + tIdx(y) + 15 * tIdx(x)) / 16 : -12 + -1 * tIdx(x) + (209 + 15 * tIdx(x)) / 16))))))); ++ji_s_tile)
        u_ij_plane[20 * (16 * ii_s_tile + tIdx(y)) + 16 * ji_s_tile + tIdx(x)] = u[65536 * (-2 + 16 * bIdx(z) + 16 * ii_s_tile + tIdx(y)) + 256 * (-2 + 16 * bIdx(y) + 16 * ji_s_tile + tIdx(x)) + ki + 32 * bIdx(x)];
    if (-2 + tIdx(x) + 16 * bIdx(y) >= 0 && 253 + -1 * tIdx(x) + -16 * bIdx(y) >= 0 && -2 + tIdx(y) + 16 * bIdx(z) >= 0 && 253 + -1 * tIdx(y) + -16 * bIdx(z) >= 0)
    {
      if (-1 + ki >= 0 && -3 + ki + 32 * bIdx(x) >= 0 && 3 + -1 * 0 >= 0)
        u_k_buf[0] = u_k_buf[1];
      if (-4 + 0 == 0)
        u_k_buf[0] = u[65536 * (tIdx(y) + 16 * bIdx(z)) + 256 * (tIdx(x) + 16 * bIdx(y)) + -2 + ki + 32 * bIdx(x)];
      if (3 + -1 * 0 >= 0)
      {
        if (-2 + ki == 0 && bIdx(x) == 0)
          u_k_buf[0] = u[65536 * (tIdx(y) + 16 * bIdx(z)) + 256 * (tIdx(x) + 16 * bIdx(y)) + -2 + ki + 32 * bIdx(x)];
        if (ki == 0)
          u_k_buf[0] = u[65536 * (tIdx(y) + 16 * bIdx(z)) + 256 * (tIdx(x) + 16 * bIdx(y)) + -2 + ki + 32 * bIdx(x)];
      }
      if (-1 + ki >= 0 && -3 + ki + 32 * bIdx(x) >= 0 && 3 + -1 * 1 >= 0)
        u_k_buf[1] = u_k_buf[2];
      if (-4 + 1 == 0)
        u_k_buf[1] = u[65536 * (tIdx(y) + 16 * bIdx(z)) + 256 * (tIdx(x) + 16 * bIdx(y)) + -2 + 1 + ki + 32 * bIdx(x)];
      if (3 + -1 * 1 >= 0)
      {
        if (-2 + ki == 0 && bIdx(x) == 0)
          u_k_buf[1] = u[65536 * (tIdx(y) + 16 * bIdx(z)) + 256 * (tIdx(x) + 16 * bIdx(y)) + -2 + 1 + ki + 32 * bIdx(x)];
        if (ki == 0)
          u_k_buf[1] = u[65536 * (tIdx(y) + 16 * bIdx(z)) + 256 * (tIdx(x) + 16 * bIdx(y)) + -2 + 1 + ki + 32 * bIdx(x)];
      }
      if (-1 + ki >= 0 && -3 + ki + 32 * bIdx(x) >= 0 && 3 + -1 * 2 >= 0)
        u_k_buf[2] = u_k_buf[3];
      if (-4 + 2 == 0)
        u_k_buf[2] = u[65536 * (tIdx(y) + 16 * bIdx(z)) + 256 * (tIdx(x) + 16 * bIdx(y)) + -2 + 2 + ki + 32 * bIdx(x)];
      if (3 + -1 * 2 >= 0)
      {
        if (-2 + ki == 0 && bIdx(x) == 0)
          u_k_buf[2] = u[65536 * (tIdx(y) + 16 * bIdx(z)) + 256 * (tIdx(x) + 16 * bIdx(y)) + -2 + 2 + ki + 32 * bIdx(x)];
        if (ki == 0)
          u_k_buf[2] = u[65536 * (tIdx(y) + 16 * bIdx(z)) + 256 * (tIdx(x) + 16 * bIdx(y)) + -2 + 2 + ki + 32 * bIdx(x)];
      }
      if (-1 + ki >= 0 && -3 + ki + 32 * bIdx(x) >= 0 && 3 + -1 * 3 >= 0)
        u_k_buf[3] = u_k_buf[4];
      if (-4 + 3 == 0)
        u_k_buf[3] = u[65536 * (tIdx(y) + 16 * bIdx(z)) + 256 * (tIdx(x) + 16 * bIdx(y)) + -2 + 3 + ki + 32 * bIdx(x)];
      if (3 + -1 * 3 >= 0)
      {
        if (-2 + ki == 0 && bIdx(x) == 0)
          u_k_buf[3] = u[65536 * (tIdx(y) + 16 * bIdx(z)) + 256 * (tIdx(x) + 16 * bIdx(y)) + -2 + 3 + ki + 32 * bIdx(x)];
        if (ki == 0)
          u_k_buf[3] = u[65536 * (tIdx(y) + 16 * bIdx(z)) + 256 * (tIdx(x) + 16 * bIdx(y)) + -2 + 3 + ki + 32 * bIdx(x)];
      }
      if (-1 + ki >= 0 && -3 + ki + 32 * bIdx(x) >= 0 && 3 + -1 * 4 >= 0)
        u_k_buf[4] = u_k_buf[5];
      if (-4 + 4 == 0)
        u_k_buf[4] = u[65536 * (tIdx(y) + 16 * bIdx(z)) + 256 * (tIdx(x) + 16 * bIdx(y)) + -2 + 4 + ki + 32 * bIdx(x)];
      if (3 + -1 * 4 >= 0)
      {
        if (-2 + ki == 0 && bIdx(x) == 0)
          u_k_buf[4] = u[65536 * (tIdx(y) + 16 * bIdx(z)) + 256 * (tIdx(x) + 16 * bIdx(y)) + -2 + 4 + ki + 32 * bIdx(x)];
        if (ki == 0)
          u_k_buf[4] = u[65536 * (tIdx(y) + 16 * bIdx(z)) + 256 * (tIdx(x) + 16 * bIdx(y)) + -2 + 4 + ki + 32 * bIdx(x)];
      }
    }
    __syncthreads() /* for u_ij_plane (insn_l_update depends on u_plane_compute) */;
    if (-2 + tIdx(x) + 16 * bIdx(y) >= 0 && 253 + -1 * tIdx(x) + -16 * bIdx(y) >= 0 && -2 + tIdx(y) + 16 * bIdx(z) >= 0 && 253 + -1 * tIdx(y) + -16 * bIdx(z) >= 0)
    {
      for (int l = -2; l <= 2; ++l)
        acc_l = acc_l + c[2 + l] * (u_ij_plane[20 * (2 + tIdx(y) + -1 * l) + 2 + tIdx(x)] + u_ij_plane[20 * (2 + tIdx(y)) + 2 + tIdx(x) + -1 * l] + u_k_buf[2 + -1 * l]);
      lap_u[65536 * (tIdx(y) + 16 * bIdx(z)) + 256 * (tIdx(x) + 16 * bIdx(y)) + ki + 32 * bIdx(x)] = acc_l;
    }
  }
}