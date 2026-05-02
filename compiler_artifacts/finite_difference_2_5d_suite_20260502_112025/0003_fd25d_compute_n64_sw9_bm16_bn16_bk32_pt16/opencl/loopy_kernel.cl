#define lid(N) ((int) get_local_id(N))
#define gid(N) ((int) get_group_id(N))
#if __OPENCL_C_VERSION__ < 120
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif
#define LOOPY_CALL_WITH_INTEGER_TYPES(MACRO_NAME) \
    MACRO_NAME(int8, char) \
    MACRO_NAME(int16, short) \
    MACRO_NAME(int32, int) \
    MACRO_NAME(int64, long)
#define LOOPY_DEFINE_FLOOR_DIV_POS_B(SUFFIX, TYPE) \
    static inline TYPE loopy_floor_div_pos_b_##SUFFIX(TYPE a, TYPE b) \
    { \
        if (a<0) \
            a = a - (b-1); \
        return a/b; \
    }
LOOPY_CALL_WITH_INTEGER_TYPES(LOOPY_DEFINE_FLOOR_DIV_POS_B)
#undef LOOPY_DEFINE_FLOOR_DIV_POS_B
#undef LOOPY_CALL_WITH_INTEGER_TYPES

__kernel void __attribute__ ((reqd_work_group_size(16, 16, 1))) loopy_kernel(__global double const *__restrict__ u, __global double *__restrict__ lap_u, __global double const *__restrict__ c)
{
  double acc_l;
  __local double u_ij_plane[24 * 24];
  double u_k_buf[9];

  for (int ki = ((-1 + gid(0) == 0) ? 0 : 4); ki <= ((-1 + gid(0) == 0) ? 27 : 31); ++ki)
  {
    if (-4 + lid(0) + 16 * gid(1) >= 0 && 59 + -1 * lid(0) + -16 * gid(1) >= 0 && -4 + lid(1) + 16 * gid(2) >= 0 && 59 + -1 * lid(1) + -16 * gid(2) >= 0)
      acc_l = (double) (0.0);
    barrier(CLK_LOCAL_MEM_FENCE) /* for u_ij_plane (u_plane_compute rev-depends on insn_l_update) */;
    for (int ii_s_tile = (((-4 + lid(1) + 16 * gid(2) >= 0 && 3 + -1 * lid(1) + -1 * lid(0) >= 0 && -4 + lid(0) + 16 * gid(1) >= 0 && 47 + lid(1) + -1 * lid(0) + -16 * gid(1) >= 0) || (-4 + lid(1) + 16 * gid(2) >= 0 && -4 + lid(1) + lid(0) >= 0 && -8 + lid(1) + 16 * gid(2) + lid(0) >= 0 && -4 + lid(0) + 16 * gid(1) >= 0 && -8 + lid(1) + lid(0) + 16 * gid(1) >= 0 && -12 + lid(1) + 16 * gid(2) + lid(0) + 16 * gid(1) >= 0) || (gid(1) == 0 && -4 + lid(1) + 16 * gid(2) >= 0 && 3 + -1 * lid(0) >= 0) || (gid(1) == 0 && -4 + lid(1) + 16 * gid(2) >= 0 && -4 + lid(0) >= 0 && 7 + -1 * lid(1) + -1 * lid(0) >= 0 && 3 + lid(1) + -1 * lid(0) >= 0) || (gid(2) == 0 && -4 + lid(1) >= 0 && 7 + -1 * lid(1) + -1 * lid(0) >= 0 && -1 + gid(1) >= 0 && 43 + lid(1) + -1 * lid(0) + -16 * gid(1) >= 0) || (gid(1) == 0 && gid(2) == 0 && -4 + lid(0) >= 0 && 11 + -1 * lid(1) + -1 * lid(0) >= 0 && -1 + lid(1) + -1 * lid(0) >= 0)) ? 0 : 1); ii_s_tile <= (((-3 + gid(2) == 0 && -8 + lid(1) >= 0 && 14 + -1 * lid(1) >= 0 && 3 + -1 * gid(1) >= 0) || (-3 + gid(2) == 0 && -4 + lid(1) >= 0 && 7 + -1 * lid(1) >= 0 && 3 + -1 * gid(1) >= 0) || (-3 + gid(1) == 0 && 7 + -1 * lid(1) >= 0 && 2 + -1 * gid(2) >= 0 && -7 + lid(1) + lid(0) >= 0 && -8 + lid(1) + 16 * gid(2) + lid(0) >= 0 && -4 + lid(1) + -1 * lid(0) >= 0) || (gid(1) == 0 && 7 + -1 * lid(1) >= 0 && 2 + -1 * gid(2) >= 0 && -11 + lid(1) + lid(0) >= 0 && -12 + lid(1) + 16 * gid(2) + lid(0) >= 0 && lid(1) + -1 * lid(0) >= 0) || (-3 + gid(1) == 0 && -3 + gid(2) == 0 && 3 + -1 * lid(1) >= 0 && -3 + lid(1) + lid(0) >= 0 && lid(1) + -1 * lid(0) >= 0) || (gid(1) == 0 && -3 + gid(2) == 0 && 3 + -1 * lid(1) >= 0 && -7 + lid(1) + lid(0) >= 0 && 4 + lid(1) + -1 * lid(0) >= 0) || (-3 + gid(1) == 0 && -7 + lid(1) + lid(0) == 0 && gid(2) == 0 && -8 + lid(1) >= 0) || (-3 + gid(1) == 0 && -7 + lid(1) + lid(0) == 0 && gid(2) == 0 && -6 + lid(1) >= 0 && 7 + -1 * lid(1) >= 0) || (gid(1) == 0 && -11 + lid(1) + lid(0) == 0 && gid(2) == 0 && -6 + lid(1) >= 0 && 7 + -1 * lid(1) >= 0)) ? 0 : (((-3 + gid(2) == 0 && 2 + -1 * gid(1) >= 0 && 1 + lid(1) + 16 * loopy_floor_div_pos_b_int32(3 + -1 * lid(1), 16) >= 0 && 3 + -1 * lid(1) + -16 * loopy_floor_div_pos_b_int32(3 + -1 * lid(1), 16) >= 0 && 11 + -1 * lid(1) + lid(0) + 16 * gid(1) + -32 * loopy_floor_div_pos_b_int32(3 + -1 * lid(1), 16) + 16 * loopy_floor_div_pos_b_int32(6 + -1 * lid(1) + -1 * lid(0), 16) >= 0 && 9 + lid(1) + lid(0) + 16 * loopy_floor_div_pos_b_int32(6 + -1 * lid(1) + -1 * lid(0), 16) >= 0 && 6 + -1 * lid(1) + -1 * lid(0) + -16 * loopy_floor_div_pos_b_int32(6 + -1 * lid(1) + -1 * lid(0), 16) >= 0) || (-3 + gid(1) == 0 && -3 + gid(2) == 0 && 1 + lid(1) + 16 * loopy_floor_div_pos_b_int32(3 + -1 * lid(1), 16) >= 0 && 3 + -1 * lid(1) + -16 * loopy_floor_div_pos_b_int32(3 + -1 * lid(1), 16) >= 0 && 15 + -1 * lid(1) + lid(0) + -32 * loopy_floor_div_pos_b_int32(3 + -1 * lid(1), 16) + 16 * loopy_floor_div_pos_b_int32(2 + -1 * lid(1) + -1 * lid(0), 16) >= 0 && 13 + lid(1) + lid(0) + 16 * loopy_floor_div_pos_b_int32(2 + -1 * lid(1) + -1 * lid(0), 16) >= 0 && 2 + -1 * lid(1) + -1 * lid(0) + -16 * loopy_floor_div_pos_b_int32(2 + -1 * lid(1) + -1 * lid(0), 16) >= 0)) ? 1 + -1 * lid(1) + (3 + 15 * lid(1)) / 16 : 1 + -1 * lid(1) + (7 + 15 * lid(1)) / 16)); ++ii_s_tile)
      for (int ji_s_tile = ((63 + -1 * lid(1) + -16 * ii_s_tile + -16 * gid(2) + lid(0) >= 0 && 19 + -1 * lid(1) + -16 * ii_s_tile + lid(0) >= 0 && -4 + lid(1) + 16 * ii_s_tile + lid(0) >= 0 && -8 + lid(1) + 16 * ii_s_tile + 16 * gid(2) + lid(0) >= 0 && 59 + -1 * lid(1) + -16 * ii_s_tile + -16 * gid(2) + lid(0) + 16 * gid(1) >= 0 && 15 + -1 * lid(1) + -16 * ii_s_tile + lid(0) + 16 * gid(1) >= 0 && -4 + lid(0) + 16 * gid(1) >= 0 && -8 + lid(1) + 16 * ii_s_tile + lid(0) + 16 * gid(1) >= 0 && -12 + lid(1) + 16 * ii_s_tile + 16 * gid(2) + lid(0) + 16 * gid(1) >= 0) ? 0 : 1); ji_s_tile <= ((-3 + gid(1) == 0 && -3 + gid(2) == 0 && -15 + lid(1) + 16 * ii_s_tile >= 0 && 19 + -1 * lid(1) + -16 * ii_s_tile >= 0 && 50 + -1 * lid(1) + -32 * ii_s_tile + -1 * lid(0) + 16 * loopy_floor_div_pos_b_int32(-1 + -1 * lid(1) + lid(0), 16) >= 0 && 16 + lid(1) + -1 * lid(0) + 16 * loopy_floor_div_pos_b_int32(-1 + -1 * lid(1) + lid(0), 16) >= 0 && -1 + -1 * lid(1) + lid(0) + -16 * loopy_floor_div_pos_b_int32(-1 + -1 * lid(1) + lid(0), 16) >= 0) ? 2 + -1 * lid(1) + -1 * ii_s_tile + -1 * lid(0) + (2 + 15 * lid(1) + 15 * lid(0)) / 16 : (((-3 + gid(1) == 0 && -1 + ii_s_tile == 0 && 2 + -1 * lid(1) >= 0 && 2 + -1 * gid(2) >= 0) || (-3 + gid(1) == 0 && ii_s_tile == 0 && -4 + lid(1) >= 0 && -8 + lid(1) + 16 * gid(2) >= 0 && 58 + -1 * lid(1) + -16 * gid(2) >= 0) || (-3 + gid(1) == 0 && -3 + gid(2) == 0 && ii_s_tile == 0 && -11 + lid(1) >= 0 && 14 + -1 * lid(1) >= 0)) ? 1 + -1 * lid(0) + (3 + 15 * lid(0)) / 16 : (((ii_s_tile == 0 && 3 + -1 * lid(1) >= 0 && -1 + gid(2) >= 0 && -1 + gid(1) >= 0 && 2 + -1 * gid(1) >= 0) || (gid(1) == 0 && ii_s_tile == 0 && 3 + -1 * lid(1) >= 0 && -1 + gid(2) >= 0 && 7 + -1 * lid(1) + -1 * lid(0) + 16 * ((8 + lid(1) + lid(0)) / 16) >= 0 && 3 + lid(1) + -1 * lid(0) + 16 * ((8 + lid(1) + lid(0)) / 16) >= 0 && 8 + lid(1) + lid(0) + -16 * ((8 + lid(1) + lid(0)) / 16) >= 0)) ? 1 + -1 * lid(0) + (3 + lid(1) + 15 * lid(0)) / 16 : (((-3 + gid(1) == 0 && -1 + ii_s_tile == 0 && -3 + lid(1) >= 0 && 7 + -1 * lid(1) >= 0 && 2 + -1 * gid(2) >= 0 && 22 + -1 * lid(1) + -1 * lid(0) + 16 * loopy_floor_div_pos_b_int32(3 + -1 * lid(1) + lid(0), 16) >= 0 && 12 + lid(1) + -1 * lid(0) + 16 * loopy_floor_div_pos_b_int32(3 + -1 * lid(1) + lid(0), 16) >= 0 && 3 + -1 * lid(1) + lid(0) + -16 * loopy_floor_div_pos_b_int32(3 + -1 * lid(1) + lid(0), 16) >= 0) || (-3 + gid(2) == 0 && -1 + ii_s_tile == 0 && 3 + -1 * lid(1) >= 0 && gid(1) >= 0 && 2 + -1 * gid(1) >= 0 && 22 + -1 * lid(1) + -1 * lid(0) + 4 * gid(1) + 16 * loopy_floor_div_pos_b_int32(-5 + -1 * lid(1) + lid(0), 16) >= 0 && 20 + lid(1) + -1 * lid(0) + 16 * loopy_floor_div_pos_b_int32(-5 + -1 * lid(1) + lid(0), 16) >= 0 && -5 + -1 * lid(1) + lid(0) + -16 * loopy_floor_div_pos_b_int32(-5 + -1 * lid(1) + lid(0), 16) >= 0)) ? 1 + -1 * lid(1) + -1 * lid(0) + (6 + 15 * lid(1) + 15 * lid(0)) / 16 : ((-4 + lid(1) + 16 * ii_s_tile >= 0 && 19 + -1 * lid(1) + -16 * ii_s_tile >= 0 && 3 + -1 * ii_s_tile + -1 * gid(2) >= 0 && -8 + lid(1) + 16 * ii_s_tile + 16 * gid(2) >= 0 && 2 + -1 * gid(1) >= 0) ? 1 + -1 * lid(0) + (7 + 15 * lid(0)) / 16 : ((-1 + ii_s_tile == 0 && -4 + lid(1) >= 0 && 7 + -1 * lid(1) >= 0 && 2 + -1 * gid(2) >= 0 && gid(1) >= 0 && 2 + -1 * gid(1) >= 0 && 26 + -1 * lid(1) + -1 * lid(0) + 4 * gid(1) + 16 * loopy_floor_div_pos_b_int32(-1 + -1 * lid(1) + lid(0), 16) >= 0 && 16 + lid(1) + -1 * lid(0) + 16 * loopy_floor_div_pos_b_int32(-1 + -1 * lid(1) + lid(0), 16) >= 0 && -1 + -1 * lid(1) + lid(0) + -16 * loopy_floor_div_pos_b_int32(-1 + -1 * lid(1) + lid(0), 16) >= 0) ? 1 + -1 * lid(1) + -1 * lid(0) + (10 + 15 * lid(1) + 15 * lid(0)) / 16 : ((-3 + gid(1) == 0 && gid(2) == 0 && ii_s_tile == 0 && -4 + lid(1) >= 0 && 7 + -1 * lid(1) >= 0 && 7 + -1 * lid(1) + -1 * lid(0) + 16 * ((8 + lid(1) + lid(0)) / 16) >= 0 && -5 + lid(1) + -1 * lid(0) + 16 * ((8 + lid(1) + lid(0)) / 16) >= 0 && 8 + lid(1) + lid(0) + -16 * ((8 + lid(1) + lid(0)) / 16) >= 0) ? -1 * lid(0) + (11 + lid(1) + 15 * lid(0)) / 16 : -1 * lid(0) + (15 + lid(1) + 15 * lid(0)) / 16))))))); ++ji_s_tile)
        u_ij_plane[24 * (16 * ii_s_tile + lid(1)) + 16 * ji_s_tile + lid(0)] = u[4096 * (-4 + 16 * gid(2) + 16 * ii_s_tile + lid(1)) + 64 * (-4 + 16 * gid(1) + 16 * ji_s_tile + lid(0)) + ki + 32 * gid(0)];
    if (-4 + lid(0) + 16 * gid(1) >= 0 && 59 + -1 * lid(0) + -16 * gid(1) >= 0 && -4 + lid(1) + 16 * gid(2) >= 0 && 59 + -1 * lid(1) + -16 * gid(2) >= 0)
    {
      if (-1 + ki >= 0 && -5 + ki + 32 * gid(0) >= 0 && 7 + -1 * 0 >= 0)
        u_k_buf[0] = u_k_buf[1];
      if (-8 + 0 == 0)
        u_k_buf[0] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + ki + 32 * gid(0)];
      if (7 + -1 * 0 >= 0)
      {
        if (-4 + ki == 0 && gid(0) == 0)
          u_k_buf[0] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + ki + 32 * gid(0)];
        if (-1 + gid(0) == 0 && ki == 0)
          u_k_buf[0] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + ki + 32 * gid(0)];
      }
      if (-1 + ki >= 0 && -5 + ki + 32 * gid(0) >= 0 && 7 + -1 * 1 >= 0)
        u_k_buf[1] = u_k_buf[2];
      if (-8 + 1 == 0)
        u_k_buf[1] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + 1 + ki + 32 * gid(0)];
      if (7 + -1 * 1 >= 0)
      {
        if (-4 + ki == 0 && gid(0) == 0)
          u_k_buf[1] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + 1 + ki + 32 * gid(0)];
        if (-1 + gid(0) == 0 && ki == 0)
          u_k_buf[1] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + 1 + ki + 32 * gid(0)];
      }
      if (-1 + ki >= 0 && -5 + ki + 32 * gid(0) >= 0 && 7 + -1 * 2 >= 0)
        u_k_buf[2] = u_k_buf[3];
      if (-8 + 2 == 0)
        u_k_buf[2] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + 2 + ki + 32 * gid(0)];
      if (7 + -1 * 2 >= 0)
      {
        if (-4 + ki == 0 && gid(0) == 0)
          u_k_buf[2] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + 2 + ki + 32 * gid(0)];
        if (-1 + gid(0) == 0 && ki == 0)
          u_k_buf[2] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + 2 + ki + 32 * gid(0)];
      }
      if (-1 + ki >= 0 && -5 + ki + 32 * gid(0) >= 0 && 7 + -1 * 3 >= 0)
        u_k_buf[3] = u_k_buf[4];
      if (-8 + 3 == 0)
        u_k_buf[3] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + 3 + ki + 32 * gid(0)];
      if (7 + -1 * 3 >= 0)
      {
        if (-4 + ki == 0 && gid(0) == 0)
          u_k_buf[3] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + 3 + ki + 32 * gid(0)];
        if (-1 + gid(0) == 0 && ki == 0)
          u_k_buf[3] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + 3 + ki + 32 * gid(0)];
      }
      if (-1 + ki >= 0 && -5 + ki + 32 * gid(0) >= 0 && 7 + -1 * 4 >= 0)
        u_k_buf[4] = u_k_buf[5];
      if (-8 + 4 == 0)
        u_k_buf[4] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + 4 + ki + 32 * gid(0)];
      if (7 + -1 * 4 >= 0)
      {
        if (-4 + ki == 0 && gid(0) == 0)
          u_k_buf[4] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + 4 + ki + 32 * gid(0)];
        if (-1 + gid(0) == 0 && ki == 0)
          u_k_buf[4] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + 4 + ki + 32 * gid(0)];
      }
      if (-1 + ki >= 0 && -5 + ki + 32 * gid(0) >= 0 && 7 + -1 * 5 >= 0)
        u_k_buf[5] = u_k_buf[6];
      if (-8 + 5 == 0)
        u_k_buf[5] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + 5 + ki + 32 * gid(0)];
      if (7 + -1 * 5 >= 0)
      {
        if (-4 + ki == 0 && gid(0) == 0)
          u_k_buf[5] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + 5 + ki + 32 * gid(0)];
        if (-1 + gid(0) == 0 && ki == 0)
          u_k_buf[5] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + 5 + ki + 32 * gid(0)];
      }
      if (-1 + ki >= 0 && -5 + ki + 32 * gid(0) >= 0 && 7 + -1 * 6 >= 0)
        u_k_buf[6] = u_k_buf[7];
      if (-8 + 6 == 0)
        u_k_buf[6] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + 6 + ki + 32 * gid(0)];
      if (7 + -1 * 6 >= 0)
      {
        if (-4 + ki == 0 && gid(0) == 0)
          u_k_buf[6] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + 6 + ki + 32 * gid(0)];
        if (-1 + gid(0) == 0 && ki == 0)
          u_k_buf[6] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + 6 + ki + 32 * gid(0)];
      }
      if (-1 + ki >= 0 && -5 + ki + 32 * gid(0) >= 0 && 7 + -1 * 7 >= 0)
        u_k_buf[7] = u_k_buf[8];
      if (-8 + 7 == 0)
        u_k_buf[7] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + 7 + ki + 32 * gid(0)];
      if (7 + -1 * 7 >= 0)
      {
        if (-4 + ki == 0 && gid(0) == 0)
          u_k_buf[7] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + 7 + ki + 32 * gid(0)];
        if (-1 + gid(0) == 0 && ki == 0)
          u_k_buf[7] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + 7 + ki + 32 * gid(0)];
      }
      if (-1 + ki >= 0 && -5 + ki + 32 * gid(0) >= 0 && 7 + -1 * 8 >= 0)
        u_k_buf[8] = u_k_buf[9];
      if (-8 + 8 == 0)
        u_k_buf[8] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + 8 + ki + 32 * gid(0)];
      if (7 + -1 * 8 >= 0)
      {
        if (-4 + ki == 0 && gid(0) == 0)
          u_k_buf[8] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + 8 + ki + 32 * gid(0)];
        if (-1 + gid(0) == 0 && ki == 0)
          u_k_buf[8] = u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + -4 + 8 + ki + 32 * gid(0)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE) /* for u_ij_plane (insn_l_update depends on u_plane_compute) */;
    if (-4 + lid(0) + 16 * gid(1) >= 0 && 59 + -1 * lid(0) + -16 * gid(1) >= 0 && -4 + lid(1) + 16 * gid(2) >= 0 && 59 + -1 * lid(1) + -16 * gid(2) >= 0)
    {
      for (int l = -4; l <= 4; ++l)
        acc_l = acc_l + c[4 + l] * (u_ij_plane[24 * (4 + lid(1) + -1 * l) + 4 + lid(0)] + u_ij_plane[24 * (4 + lid(1)) + 4 + lid(0) + -1 * l] + u_k_buf[4 + -1 * l]);
      lap_u[4096 * (lid(1) + 16 * gid(2)) + 64 * (lid(0) + 16 * gid(1)) + ki + 32 * gid(0)] = acc_l;
    }
  }
}