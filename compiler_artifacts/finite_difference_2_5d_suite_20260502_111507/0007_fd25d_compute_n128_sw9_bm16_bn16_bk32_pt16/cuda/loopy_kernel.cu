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
  __shared__ double u_ij_plane[24 * 24];
  double u_k_buf[9];

  for (int ki = ((-1 + bIdx(x) >= 0) ? 0 : 4); ki <= ((-3 + bIdx(x) == 0) ? 27 : 31); ++ki)
  {
    if (-4 + tIdx(x) + 16 * bIdx(y) >= 0 && 123 + -1 * tIdx(x) + -16 * bIdx(y) >= 0 && -4 + tIdx(y) + 16 * bIdx(z) >= 0 && 123 + -1 * tIdx(y) + -16 * bIdx(z) >= 0)
      acc_l = (double) (0.0);
    __syncthreads() /* for u_ij_plane (u_plane_compute rev-depends on insn_l_update) */;
    for (int ii_s_tile = (((-4 + tIdx(y) + 16 * bIdx(z) >= 0 && 3 + -1 * tIdx(y) + -1 * tIdx(x) >= 0 && -4 + tIdx(x) + 16 * bIdx(y) >= 0 && 111 + tIdx(y) + -1 * tIdx(x) + -16 * bIdx(y) >= 0) || (-4 + tIdx(y) + 16 * bIdx(z) >= 0 && -4 + tIdx(y) + tIdx(x) >= 0 && -8 + tIdx(y) + 16 * bIdx(z) + tIdx(x) >= 0 && -4 + tIdx(x) + 16 * bIdx(y) >= 0 && -8 + tIdx(y) + tIdx(x) + 16 * bIdx(y) >= 0 && -12 + tIdx(y) + 16 * bIdx(z) + tIdx(x) + 16 * bIdx(y) >= 0) || (bIdx(y) == 0 && -4 + tIdx(y) + 16 * bIdx(z) >= 0 && 3 + -1 * tIdx(x) >= 0) || (bIdx(y) == 0 && -4 + tIdx(y) + 16 * bIdx(z) >= 0 && -4 + tIdx(x) >= 0 && 7 + -1 * tIdx(y) + -1 * tIdx(x) >= 0 && 3 + tIdx(y) + -1 * tIdx(x) >= 0) || (bIdx(z) == 0 && -4 + tIdx(y) >= 0 && 7 + -1 * tIdx(y) + -1 * tIdx(x) >= 0 && -1 + bIdx(y) >= 0 && 107 + tIdx(y) + -1 * tIdx(x) + -16 * bIdx(y) >= 0) || (bIdx(y) == 0 && bIdx(z) == 0 && -4 + tIdx(x) >= 0 && 11 + -1 * tIdx(y) + -1 * tIdx(x) >= 0 && -1 + tIdx(y) + -1 * tIdx(x) >= 0)) ? 0 : 1); ii_s_tile <= (((-7 + bIdx(z) == 0 && -8 + tIdx(y) >= 0 && 14 + -1 * tIdx(y) >= 0 && 7 + -1 * bIdx(y) >= 0) || (-7 + bIdx(z) == 0 && -4 + tIdx(y) >= 0 && 7 + -1 * tIdx(y) >= 0 && 7 + -1 * bIdx(y) >= 0) || (-7 + bIdx(y) == 0 && 7 + -1 * tIdx(y) >= 0 && 6 + -1 * bIdx(z) >= 0 && -7 + tIdx(y) + tIdx(x) >= 0 && -8 + tIdx(y) + 16 * bIdx(z) + tIdx(x) >= 0 && -4 + tIdx(y) + -1 * tIdx(x) >= 0) || (bIdx(y) == 0 && 7 + -1 * tIdx(y) >= 0 && 6 + -1 * bIdx(z) >= 0 && -11 + tIdx(y) + tIdx(x) >= 0 && -12 + tIdx(y) + 16 * bIdx(z) + tIdx(x) >= 0 && tIdx(y) + -1 * tIdx(x) >= 0) || (-7 + bIdx(y) == 0 && -7 + bIdx(z) == 0 && 3 + -1 * tIdx(y) >= 0 && -3 + tIdx(y) + tIdx(x) >= 0 && tIdx(y) + -1 * tIdx(x) >= 0) || (bIdx(y) == 0 && -7 + bIdx(z) == 0 && 3 + -1 * tIdx(y) >= 0 && -7 + tIdx(y) + tIdx(x) >= 0 && 4 + tIdx(y) + -1 * tIdx(x) >= 0) || (-7 + bIdx(y) == 0 && -7 + tIdx(y) + tIdx(x) == 0 && bIdx(z) == 0 && -8 + tIdx(y) >= 0) || (-7 + bIdx(y) == 0 && -7 + tIdx(y) + tIdx(x) == 0 && bIdx(z) == 0 && -6 + tIdx(y) >= 0 && 7 + -1 * tIdx(y) >= 0) || (bIdx(y) == 0 && -11 + tIdx(y) + tIdx(x) == 0 && bIdx(z) == 0 && -6 + tIdx(y) >= 0 && 7 + -1 * tIdx(y) >= 0)) ? 0 : (((-7 + bIdx(z) == 0 && 6 + -1 * bIdx(y) >= 0 && 1 + tIdx(y) + 16 * loopy_floor_div_pos_b_int32(3 + -1 * tIdx(y), 16) >= 0 && 3 + -1 * tIdx(y) + -16 * loopy_floor_div_pos_b_int32(3 + -1 * tIdx(y), 16) >= 0 && 11 + -1 * tIdx(y) + tIdx(x) + 16 * bIdx(y) + -32 * loopy_floor_div_pos_b_int32(3 + -1 * tIdx(y), 16) + 16 * loopy_floor_div_pos_b_int32(6 + -1 * tIdx(y) + -1 * tIdx(x), 16) >= 0 && 9 + tIdx(y) + tIdx(x) + 16 * loopy_floor_div_pos_b_int32(6 + -1 * tIdx(y) + -1 * tIdx(x), 16) >= 0 && 6 + -1 * tIdx(y) + -1 * tIdx(x) + -16 * loopy_floor_div_pos_b_int32(6 + -1 * tIdx(y) + -1 * tIdx(x), 16) >= 0) || (-7 + bIdx(y) == 0 && -7 + bIdx(z) == 0 && 1 + tIdx(y) + 16 * loopy_floor_div_pos_b_int32(3 + -1 * tIdx(y), 16) >= 0 && 3 + -1 * tIdx(y) + -16 * loopy_floor_div_pos_b_int32(3 + -1 * tIdx(y), 16) >= 0 && 15 + -1 * tIdx(y) + tIdx(x) + -32 * loopy_floor_div_pos_b_int32(3 + -1 * tIdx(y), 16) + 16 * loopy_floor_div_pos_b_int32(2 + -1 * tIdx(y) + -1 * tIdx(x), 16) >= 0 && 13 + tIdx(y) + tIdx(x) + 16 * loopy_floor_div_pos_b_int32(2 + -1 * tIdx(y) + -1 * tIdx(x), 16) >= 0 && 2 + -1 * tIdx(y) + -1 * tIdx(x) + -16 * loopy_floor_div_pos_b_int32(2 + -1 * tIdx(y) + -1 * tIdx(x), 16) >= 0)) ? 1 + -1 * tIdx(y) + (3 + 15 * tIdx(y)) / 16 : 1 + -1 * tIdx(y) + (7 + 15 * tIdx(y)) / 16)); ++ii_s_tile)
      for (int ji_s_tile = ((127 + -1 * tIdx(y) + -16 * ii_s_tile + -16 * bIdx(z) + tIdx(x) >= 0 && 19 + -1 * tIdx(y) + -16 * ii_s_tile + tIdx(x) >= 0 && -4 + tIdx(y) + 16 * ii_s_tile + tIdx(x) >= 0 && -8 + tIdx(y) + 16 * ii_s_tile + 16 * bIdx(z) + tIdx(x) >= 0 && 123 + -1 * tIdx(y) + -16 * ii_s_tile + -16 * bIdx(z) + tIdx(x) + 16 * bIdx(y) >= 0 && 15 + -1 * tIdx(y) + -16 * ii_s_tile + tIdx(x) + 16 * bIdx(y) >= 0 && -4 + tIdx(x) + 16 * bIdx(y) >= 0 && -8 + tIdx(y) + 16 * ii_s_tile + tIdx(x) + 16 * bIdx(y) >= 0 && -12 + tIdx(y) + 16 * ii_s_tile + 16 * bIdx(z) + tIdx(x) + 16 * bIdx(y) >= 0) ? 0 : 1); ji_s_tile <= ((-7 + bIdx(y) == 0 && -7 + bIdx(z) == 0 && -15 + tIdx(y) + 16 * ii_s_tile >= 0 && 19 + -1 * tIdx(y) + -16 * ii_s_tile >= 0 && 50 + -1 * tIdx(y) + -32 * ii_s_tile + -1 * tIdx(x) + 16 * loopy_floor_div_pos_b_int32(-1 + -1 * tIdx(y) + tIdx(x), 16) >= 0 && 16 + tIdx(y) + -1 * tIdx(x) + 16 * loopy_floor_div_pos_b_int32(-1 + -1 * tIdx(y) + tIdx(x), 16) >= 0 && -1 + -1 * tIdx(y) + tIdx(x) + -16 * loopy_floor_div_pos_b_int32(-1 + -1 * tIdx(y) + tIdx(x), 16) >= 0) ? 2 + -1 * tIdx(y) + -1 * ii_s_tile + -1 * tIdx(x) + (2 + 15 * tIdx(y) + 15 * tIdx(x)) / 16 : (((-7 + bIdx(y) == 0 && -1 + ii_s_tile == 0 && 2 + -1 * tIdx(y) >= 0 && 6 + -1 * bIdx(z) >= 0) || (-7 + bIdx(y) == 0 && ii_s_tile == 0 && -4 + tIdx(y) >= 0 && -8 + tIdx(y) + 16 * bIdx(z) >= 0 && 122 + -1 * tIdx(y) + -16 * bIdx(z) >= 0) || (-7 + bIdx(y) == 0 && -7 + bIdx(z) == 0 && ii_s_tile == 0 && -11 + tIdx(y) >= 0 && 14 + -1 * tIdx(y) >= 0)) ? 1 + -1 * tIdx(x) + (3 + 15 * tIdx(x)) / 16 : (((ii_s_tile == 0 && 3 + -1 * tIdx(y) >= 0 && -1 + bIdx(z) >= 0 && -1 + bIdx(y) >= 0 && 6 + -1 * bIdx(y) >= 0) || (bIdx(y) == 0 && ii_s_tile == 0 && 3 + -1 * tIdx(y) >= 0 && -1 + bIdx(z) >= 0 && 7 + -1 * tIdx(y) + -1 * tIdx(x) + 16 * ((8 + tIdx(y) + tIdx(x)) / 16) >= 0 && 3 + tIdx(y) + -1 * tIdx(x) + 16 * ((8 + tIdx(y) + tIdx(x)) / 16) >= 0 && 8 + tIdx(y) + tIdx(x) + -16 * ((8 + tIdx(y) + tIdx(x)) / 16) >= 0)) ? 1 + -1 * tIdx(x) + (3 + tIdx(y) + 15 * tIdx(x)) / 16 : (((-7 + bIdx(y) == 0 && -1 + ii_s_tile == 0 && -3 + tIdx(y) >= 0 && 7 + -1 * tIdx(y) >= 0 && 6 + -1 * bIdx(z) >= 0 && 22 + -1 * tIdx(y) + -1 * tIdx(x) + 16 * loopy_floor_div_pos_b_int32(3 + -1 * tIdx(y) + tIdx(x), 16) >= 0 && 12 + tIdx(y) + -1 * tIdx(x) + 16 * loopy_floor_div_pos_b_int32(3 + -1 * tIdx(y) + tIdx(x), 16) >= 0 && 3 + -1 * tIdx(y) + tIdx(x) + -16 * loopy_floor_div_pos_b_int32(3 + -1 * tIdx(y) + tIdx(x), 16) >= 0) || (-7 + bIdx(z) == 0 && -1 + ii_s_tile == 0 && 3 + -1 * tIdx(y) >= 0 && bIdx(y) >= 0 && 6 + -1 * bIdx(y) >= 0 && 22 + -1 * tIdx(y) + -1 * tIdx(x) + 4 * bIdx(y) + 16 * loopy_floor_div_pos_b_int32(-5 + -1 * tIdx(y) + tIdx(x), 16) >= 0 && 20 + tIdx(y) + -1 * tIdx(x) + 16 * loopy_floor_div_pos_b_int32(-5 + -1 * tIdx(y) + tIdx(x), 16) >= 0 && -5 + -1 * tIdx(y) + tIdx(x) + -16 * loopy_floor_div_pos_b_int32(-5 + -1 * tIdx(y) + tIdx(x), 16) >= 0)) ? 1 + -1 * tIdx(y) + -1 * tIdx(x) + (6 + 15 * tIdx(y) + 15 * tIdx(x)) / 16 : ((-4 + tIdx(y) + 16 * ii_s_tile >= 0 && 19 + -1 * tIdx(y) + -16 * ii_s_tile >= 0 && 7 + -1 * ii_s_tile + -1 * bIdx(z) >= 0 && -8 + tIdx(y) + 16 * ii_s_tile + 16 * bIdx(z) >= 0 && 6 + -1 * bIdx(y) >= 0) ? 1 + -1 * tIdx(x) + (7 + 15 * tIdx(x)) / 16 : ((-1 + ii_s_tile == 0 && -4 + tIdx(y) >= 0 && 7 + -1 * tIdx(y) >= 0 && 6 + -1 * bIdx(z) >= 0 && bIdx(y) >= 0 && 6 + -1 * bIdx(y) >= 0 && 26 + -1 * tIdx(y) + -1 * tIdx(x) + 4 * bIdx(y) + 16 * loopy_floor_div_pos_b_int32(-1 + -1 * tIdx(y) + tIdx(x), 16) >= 0 && 16 + tIdx(y) + -1 * tIdx(x) + 16 * loopy_floor_div_pos_b_int32(-1 + -1 * tIdx(y) + tIdx(x), 16) >= 0 && -1 + -1 * tIdx(y) + tIdx(x) + -16 * loopy_floor_div_pos_b_int32(-1 + -1 * tIdx(y) + tIdx(x), 16) >= 0) ? 1 + -1 * tIdx(y) + -1 * tIdx(x) + (10 + 15 * tIdx(y) + 15 * tIdx(x)) / 16 : ((-7 + bIdx(y) == 0 && bIdx(z) == 0 && ii_s_tile == 0 && -4 + tIdx(y) >= 0 && 7 + -1 * tIdx(y) >= 0 && 7 + -1 * tIdx(y) + -1 * tIdx(x) + 16 * ((8 + tIdx(y) + tIdx(x)) / 16) >= 0 && -5 + tIdx(y) + -1 * tIdx(x) + 16 * ((8 + tIdx(y) + tIdx(x)) / 16) >= 0 && 8 + tIdx(y) + tIdx(x) + -16 * ((8 + tIdx(y) + tIdx(x)) / 16) >= 0) ? -1 * tIdx(x) + (11 + tIdx(y) + 15 * tIdx(x)) / 16 : -1 * tIdx(x) + (15 + tIdx(y) + 15 * tIdx(x)) / 16))))))); ++ji_s_tile)
        u_ij_plane[24 * (16 * ii_s_tile + tIdx(y)) + 16 * ji_s_tile + tIdx(x)] = u[16384 * (-4 + 16 * bIdx(z) + 16 * ii_s_tile + tIdx(y)) + 128 * (-4 + 16 * bIdx(y) + 16 * ji_s_tile + tIdx(x)) + ki + 32 * bIdx(x)];
    if (-4 + tIdx(x) + 16 * bIdx(y) >= 0 && 123 + -1 * tIdx(x) + -16 * bIdx(y) >= 0 && -4 + tIdx(y) + 16 * bIdx(z) >= 0 && 123 + -1 * tIdx(y) + -16 * bIdx(z) >= 0)
    {
      if (-1 + ki >= 0 && -5 + ki + 32 * bIdx(x) >= 0 && 7 + -1 * 0 >= 0)
        u_k_buf[0] = u_k_buf[1];
      if (-8 + 0 == 0)
        u_k_buf[0] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + ki + 32 * bIdx(x)];
      if (7 + -1 * 0 >= 0)
      {
        if (-4 + ki == 0 && bIdx(x) == 0)
          u_k_buf[0] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + ki + 32 * bIdx(x)];
        if (ki == 0)
          u_k_buf[0] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + ki + 32 * bIdx(x)];
      }
      if (-1 + ki >= 0 && -5 + ki + 32 * bIdx(x) >= 0 && 7 + -1 * 1 >= 0)
        u_k_buf[1] = u_k_buf[2];
      if (-8 + 1 == 0)
        u_k_buf[1] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + 1 + ki + 32 * bIdx(x)];
      if (7 + -1 * 1 >= 0)
      {
        if (-4 + ki == 0 && bIdx(x) == 0)
          u_k_buf[1] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + 1 + ki + 32 * bIdx(x)];
        if (ki == 0)
          u_k_buf[1] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + 1 + ki + 32 * bIdx(x)];
      }
      if (-1 + ki >= 0 && -5 + ki + 32 * bIdx(x) >= 0 && 7 + -1 * 2 >= 0)
        u_k_buf[2] = u_k_buf[3];
      if (-8 + 2 == 0)
        u_k_buf[2] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + 2 + ki + 32 * bIdx(x)];
      if (7 + -1 * 2 >= 0)
      {
        if (-4 + ki == 0 && bIdx(x) == 0)
          u_k_buf[2] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + 2 + ki + 32 * bIdx(x)];
        if (ki == 0)
          u_k_buf[2] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + 2 + ki + 32 * bIdx(x)];
      }
      if (-1 + ki >= 0 && -5 + ki + 32 * bIdx(x) >= 0 && 7 + -1 * 3 >= 0)
        u_k_buf[3] = u_k_buf[4];
      if (-8 + 3 == 0)
        u_k_buf[3] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + 3 + ki + 32 * bIdx(x)];
      if (7 + -1 * 3 >= 0)
      {
        if (-4 + ki == 0 && bIdx(x) == 0)
          u_k_buf[3] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + 3 + ki + 32 * bIdx(x)];
        if (ki == 0)
          u_k_buf[3] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + 3 + ki + 32 * bIdx(x)];
      }
      if (-1 + ki >= 0 && -5 + ki + 32 * bIdx(x) >= 0 && 7 + -1 * 4 >= 0)
        u_k_buf[4] = u_k_buf[5];
      if (-8 + 4 == 0)
        u_k_buf[4] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + 4 + ki + 32 * bIdx(x)];
      if (7 + -1 * 4 >= 0)
      {
        if (-4 + ki == 0 && bIdx(x) == 0)
          u_k_buf[4] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + 4 + ki + 32 * bIdx(x)];
        if (ki == 0)
          u_k_buf[4] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + 4 + ki + 32 * bIdx(x)];
      }
      if (-1 + ki >= 0 && -5 + ki + 32 * bIdx(x) >= 0 && 7 + -1 * 5 >= 0)
        u_k_buf[5] = u_k_buf[6];
      if (-8 + 5 == 0)
        u_k_buf[5] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + 5 + ki + 32 * bIdx(x)];
      if (7 + -1 * 5 >= 0)
      {
        if (-4 + ki == 0 && bIdx(x) == 0)
          u_k_buf[5] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + 5 + ki + 32 * bIdx(x)];
        if (ki == 0)
          u_k_buf[5] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + 5 + ki + 32 * bIdx(x)];
      }
      if (-1 + ki >= 0 && -5 + ki + 32 * bIdx(x) >= 0 && 7 + -1 * 6 >= 0)
        u_k_buf[6] = u_k_buf[7];
      if (-8 + 6 == 0)
        u_k_buf[6] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + 6 + ki + 32 * bIdx(x)];
      if (7 + -1 * 6 >= 0)
      {
        if (-4 + ki == 0 && bIdx(x) == 0)
          u_k_buf[6] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + 6 + ki + 32 * bIdx(x)];
        if (ki == 0)
          u_k_buf[6] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + 6 + ki + 32 * bIdx(x)];
      }
      if (-1 + ki >= 0 && -5 + ki + 32 * bIdx(x) >= 0 && 7 + -1 * 7 >= 0)
        u_k_buf[7] = u_k_buf[8];
      if (-8 + 7 == 0)
        u_k_buf[7] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + 7 + ki + 32 * bIdx(x)];
      if (7 + -1 * 7 >= 0)
      {
        if (-4 + ki == 0 && bIdx(x) == 0)
          u_k_buf[7] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + 7 + ki + 32 * bIdx(x)];
        if (ki == 0)
          u_k_buf[7] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + 7 + ki + 32 * bIdx(x)];
      }
      if (-1 + ki >= 0 && -5 + ki + 32 * bIdx(x) >= 0 && 7 + -1 * 8 >= 0)
        u_k_buf[8] = u_k_buf[9];
      if (-8 + 8 == 0)
        u_k_buf[8] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + 8 + ki + 32 * bIdx(x)];
      if (7 + -1 * 8 >= 0)
      {
        if (-4 + ki == 0 && bIdx(x) == 0)
          u_k_buf[8] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + 8 + ki + 32 * bIdx(x)];
        if (ki == 0)
          u_k_buf[8] = u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + -4 + 8 + ki + 32 * bIdx(x)];
      }
    }
    __syncthreads() /* for u_ij_plane (insn_l_update depends on u_plane_compute) */;
    if (-4 + tIdx(x) + 16 * bIdx(y) >= 0 && 123 + -1 * tIdx(x) + -16 * bIdx(y) >= 0 && -4 + tIdx(y) + 16 * bIdx(z) >= 0 && 123 + -1 * tIdx(y) + -16 * bIdx(z) >= 0)
    {
      for (int l = -4; l <= 4; ++l)
        acc_l = acc_l + c[4 + l] * (u_ij_plane[24 * (4 + tIdx(y) + -1 * l) + 4 + tIdx(x)] + u_ij_plane[24 * (4 + tIdx(y)) + 4 + tIdx(x) + -1 * l] + u_k_buf[4 + -1 * l]);
      lap_u[16384 * (tIdx(y) + 16 * bIdx(z)) + 128 * (tIdx(x) + 16 * bIdx(y)) + ki + 32 * bIdx(x)] = acc_l;
    }
  }
}