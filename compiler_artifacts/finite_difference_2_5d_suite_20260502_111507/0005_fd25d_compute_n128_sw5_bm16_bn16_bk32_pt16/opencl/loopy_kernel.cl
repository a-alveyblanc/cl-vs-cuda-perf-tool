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
  __local double u_ij_plane[20 * 20];
  double u_k_buf[5];

  for (int ki = ((-1 + gid(0) >= 0) ? 0 : 2); ki <= ((-3 + gid(0) == 0) ? 29 : 31); ++ki)
  {
    if (-2 + lid(0) + 16 * gid(1) >= 0 && 125 + -1 * lid(0) + -16 * gid(1) >= 0 && -2 + lid(1) + 16 * gid(2) >= 0 && 125 + -1 * lid(1) + -16 * gid(2) >= 0)
      acc_l = (double) (0.0);
    barrier(CLK_LOCAL_MEM_FENCE) /* for u_ij_plane (u_plane_compute rev-depends on insn_l_update) */;
    for (int ii_s_tile = (((-26 + 13 * lid(1) + 240 * gid(2) >= 0 && 1 + -1 * lid(1) + -1 * lid(0) >= 0 && -1 + gid(1) >= 0 && 111 + lid(1) + -1 * lid(0) + -16 * gid(1) >= 0) || (-26 + 13 * lid(1) + 240 * gid(2) >= 0 && -2 + lid(1) + lid(0) >= 0 && -4 + lid(1) + 16 * gid(2) + lid(0) >= 0 && -4 + lid(1) + lid(0) + 16 * gid(1) >= 0 && -6 + lid(1) + 16 * gid(2) + lid(0) + 16 * gid(1) >= 0 && -26 + 13 * lid(0) + 240 * gid(1) >= 0) || (gid(1) == 0 && -26 + 13 * lid(1) + 240 * gid(2) >= 0 && 1 + -1 * lid(1) + -1 * lid(0) >= 0) || (gid(1) == 0 && 125 + -1 * lid(1) + -16 * gid(2) + lid(0) >= 0 && -4 + lid(1) + lid(0) >= 0 && -6 + lid(1) + 16 * gid(2) + lid(0) >= 0 && 1 + -1 * lid(0) >= 0) || (gid(2) == 0 && -2 + lid(1) >= 0 && 3 + -1 * lid(1) + -1 * lid(0) >= 0 && gid(1) >= 0 && 109 + lid(1) + -1 * lid(0) + -16 * gid(1) >= 0) || (gid(1) == 0 && -1 + gid(2) >= 0 && lid(0) >= 0 && -2 + lid(1) + lid(0) >= 0 && 3 + -1 * lid(1) + -1 * lid(0) >= 0 && 1 + lid(1) + -1 * lid(0) >= 0) || (gid(1) == 0 && -7 + gid(2) == 0 && -14 + lid(1) + -1 * lid(0) >= 0) || (gid(1) == 0 && gid(2) == 0 && lid(0) >= 0 && -4 + lid(1) + lid(0) >= 0 && 5 + -1 * lid(1) + -1 * lid(0) >= 0 && -1 + lid(1) + -1 * lid(0) >= 0)) ? 0 : 1); ii_s_tile <= (((-7 + gid(2) == 0 && -4 + lid(1) >= 0 && 14 + -1 * lid(1) >= 0 && -1 + -1 * gid(1) >= 0 && 25 + -13 * lid(0) + -240 * gid(1) >= 0) || (-7 + gid(2) == 0 && -4 + lid(1) >= 0 && 14 + -1 * lid(1) >= 0 && 6 + -1 * gid(1) >= 0 && -26 + 13 * lid(0) + 240 * gid(1) >= 0) || (-7 + gid(2) == 0 && -2 + lid(1) >= 0 && 3 + -1 * lid(1) >= 0 && -1 + -1 * gid(1) >= 0 && 25 + -13 * lid(0) + -240 * gid(1) >= 0) || (-7 + gid(2) == 0 && -2 + lid(1) >= 0 && 3 + -1 * lid(1) >= 0 && 6 + -1 * gid(1) >= 0 && -26 + 13 * lid(0) + 240 * gid(1) >= 0) || (-7 + gid(2) == 0 && -4 + lid(1) >= 0 && 14 + -1 * lid(1) >= 0 && gid(1) >= 0 && 6 + -1 * gid(1) >= 0 && 25 + -13 * lid(0) + -240 * gid(1) >= 0) || (-7 + gid(2) == 0 && -2 + lid(1) >= 0 && 3 + -1 * lid(1) >= 0 && gid(1) >= 0 && 6 + -1 * gid(1) >= 0 && 25 + -13 * lid(0) + -240 * gid(1) >= 0) || (-7 + gid(1) == 0 && -7 + gid(2) == 0 && -4 + lid(1) >= 0 && 14 + -1 * lid(1) >= 0) || (-7 + gid(1) == 0 && -7 + gid(2) == 0 && -2 + lid(1) >= 0 && 3 + -1 * lid(1) >= 0) || (-7 + gid(1) == 0 && -3 + lid(1) == 0 && -1 + -1 * gid(2) >= 0 && 1 + -1 * lid(0) >= 0 && -240 * gid(2) + 13 * lid(0) >= 0) || (gid(1) == 0 && -3 + lid(1) == 0 && -1 + -1 * gid(2) >= 0 && -2 + lid(0) >= 0 && 3 + -1 * lid(0) >= 0) || (gid(1) == 0 && -3 + lid(1) == 0 && 6 + -1 * gid(2) >= 0 && -2 + lid(0) >= 0 && 3 + -1 * lid(0) >= 0 && -1 + 240 * gid(2) + -13 * lid(0) >= 0) || (-7 + gid(1) == 0 && -3 + lid(1) == 0 && -1 + -1 * gid(2) >= 0 && -1 + 240 * gid(2) + -13 * lid(0) >= 0 && 16 + lid(0) + 16 * loopy_floor_div_pos_b_int32(-1 + -1 * lid(0), 16) >= 0 && -15 + -1 * lid(0) + -16 * loopy_floor_div_pos_b_int32(-1 + -1 * lid(0), 16) >= 0) || (-7 + gid(1) == 0 && -3 + lid(1) == 0 && gid(2) >= 0 && 6 + -1 * gid(2) >= 0 && 1 + -1 * lid(0) >= 0 && -1 + 240 * gid(2) + -13 * lid(0) >= 0 && 16 + lid(0) + 16 * loopy_floor_div_pos_b_int32(-1 + -1 * lid(0), 16) >= 0 && -15 + -1 * lid(0) + -16 * loopy_floor_div_pos_b_int32(-1 + -1 * lid(0), 16) >= 0) || (-7 + gid(1) == 0 && gid(2) == 0 && -3 + lid(1) == 0 && lid(0) >= 0 && 1 + -1 * lid(0) >= 0) || (gid(1) == 0 && -7 + gid(2) == 0 && -1 + lid(1) == 0 && -2 + lid(0) >= 0 && 3 + -1 * lid(0) >= 0) || (gid(1) == 0 && gid(2) == 0 && -3 + lid(1) == 0 && -2 + lid(0) >= 0 && 3 + -1 * lid(0) >= 0) || (-7 + gid(1) == 0 && -7 + gid(2) == 0 && -1 + lid(1) == 0 && 1 + -1 * lid(0) >= 0 && 16 + lid(0) + 16 * loopy_floor_div_pos_b_int32(-1 + -1 * lid(0), 16) >= 0 && -15 + -1 * lid(0) + -16 * loopy_floor_div_pos_b_int32(-1 + -1 * lid(0), 16) >= 0)) ? 0 : (((-4 + lid(1) >= 0 && 6 + -1 * gid(2) >= 0 && -26 + 13 * lid(1) + 240 * gid(2) >= 0 && 6 + -1 * gid(1) >= 0 && -26 + 13 * lid(0) + 240 * gid(1) >= 0 && 12 + lid(1) + 16 * loopy_floor_div_pos_b_int32(3 + -1 * lid(1), 16) >= 0 && -1 * lid(1) + -16 * loopy_floor_div_pos_b_int32(3 + -1 * lid(1), 16) >= 0) || (3 + -1 * lid(1) >= 0 && 6 + -1 * gid(2) >= 0 && -26 + 13 * lid(1) + 240 * gid(2) >= 0 && 6 + -1 * gid(1) >= 0 && -26 + 13 * lid(0) + 240 * gid(1) >= 0 && 12 + lid(1) + 16 * loopy_floor_div_pos_b_int32(3 + -1 * lid(1), 16) >= 0 && -1 * lid(1) + -16 * loopy_floor_div_pos_b_int32(3 + -1 * lid(1), 16) >= 0) || (-1 + lid(1) >= 0 && 3 + -1 * lid(1) >= 0 && 6 + -1 * gid(2) >= 0 && -1 + 240 * gid(2) + -13 * lid(0) >= 0 && -1 + gid(1) >= 0 && 7 + -1 * gid(1) >= 0 && 31 + -1 * lid(1) + lid(0) + -2 * gid(1) + 16 * loopy_floor_div_pos_b_int32(2 + -1 * lid(1) + -1 * lid(0), 16) >= 0 && 13 + lid(1) + lid(0) + 16 * loopy_floor_div_pos_b_int32(2 + -1 * lid(1) + -1 * lid(0), 16) >= 0 && 2 + -1 * lid(1) + -1 * lid(0) + -16 * loopy_floor_div_pos_b_int32(2 + -1 * lid(1) + -1 * lid(0), 16) >= 0) || (gid(1) == 0 && -1 + lid(1) >= 0 && 3 + -1 * lid(1) >= 0 && 6 + -1 * gid(2) >= 0 && -11 + lid(0) >= 0 && -1 + 240 * gid(2) + -13 * lid(0) >= 0) || (gid(2) == 0 && -1 + lid(1) >= 0 && 3 + -1 * lid(1) >= 0 && -2 + lid(0) >= 0 && 7 + -1 * gid(1) >= 0 && -4 + lid(0) + 16 * gid(1) >= 0) || (-7 + gid(1) == 0 && -4 + lid(1) >= 0 && 6 + -1 * gid(2) >= 0 && -26 + 13 * lid(1) + 240 * gid(2) >= 0 && 12 + lid(1) + 16 * loopy_floor_div_pos_b_int32(3 + -1 * lid(1), 16) >= 0 && -1 * lid(1) + -16 * loopy_floor_div_pos_b_int32(3 + -1 * lid(1), 16) >= 0) || (-7 + gid(1) == 0 && 3 + -1 * lid(1) >= 0 && 6 + -1 * gid(2) >= 0 && -26 + 13 * lid(1) + 240 * gid(2) >= 0 && 12 + lid(1) + 16 * loopy_floor_div_pos_b_int32(3 + -1 * lid(1), 16) >= 0 && -1 * lid(1) + -16 * loopy_floor_div_pos_b_int32(3 + -1 * lid(1), 16) >= 0) || (gid(2) == 0 && -1 + lid(1) >= 0 && 3 + -1 * lid(1) >= 0 && 1 + -1 * lid(0) >= 0 && gid(1) >= 0 && 9 + -1 * lid(1) + -1 * gid(1) >= 0 && 7 + -1 * gid(1) >= 0) || (gid(1) == 0 && -4 + lid(1) >= 0 && 6 + -1 * gid(2) >= 0 && -26 + 13 * lid(1) + 240 * gid(2) >= 0 && 1 + -1 * lid(0) >= 0 && 12 + lid(1) + 16 * loopy_floor_div_pos_b_int32(3 + -1 * lid(1), 16) >= 0 && -1 * lid(1) + -16 * loopy_floor_div_pos_b_int32(3 + -1 * lid(1), 16) >= 0) || (gid(1) == 0 && 3 + -1 * lid(1) >= 0 && 6 + -1 * gid(2) >= 0 && -26 + 13 * lid(1) + 240 * gid(2) >= 0 && 1 + -1 * lid(0) >= 0 && 12 + lid(1) + 16 * loopy_floor_div_pos_b_int32(3 + -1 * lid(1), 16) >= 0 && -1 * lid(1) + -16 * loopy_floor_div_pos_b_int32(3 + -1 * lid(1), 16) >= 0) || (gid(1) == 0 && -1 + lid(1) >= 0 && 3 + -1 * lid(1) >= 0 && 6 + -1 * gid(2) >= 0 && 10 + -1 * lid(0) >= 0 && -1 + 240 * gid(2) + -13 * lid(0) >= 0 && 15 + -1 * lid(1) + lid(0) + 16 * loopy_floor_div_pos_b_int32(4 + -1 * lid(1) + -1 * lid(0), 16) >= 0 && 11 + lid(1) + lid(0) + 16 * loopy_floor_div_pos_b_int32(4 + -1 * lid(1) + -1 * lid(0), 16) >= 0 && 4 + -1 * lid(1) + -1 * lid(0) + -16 * loopy_floor_div_pos_b_int32(4 + -1 * lid(1) + -1 * lid(0), 16) >= 0) || (gid(2) == 0 && lid(1) == 0 && 6 + -1 * gid(1) >= 0 && -4 + lid(0) + 16 * gid(1) >= 0) || (gid(1) == 0 && gid(2) == 0 && -1 + lid(1) >= 0 && 2 + -1 * lid(1) >= 0 && -2 + lid(0) >= 0 && 3 + -1 * lid(0) >= 0) || (-7 + gid(1) == 0 && gid(2) == 0 && lid(1) == 0) || (gid(1) == 0 && gid(2) == 0 && lid(1) == 0 && 3 + -1 * lid(0) >= 0)) ? 1 + -1 * lid(1) + (3 + 15 * lid(1)) / 16 : -5 + -1 * lid(1) + (97 + 15 * lid(1)) / 16)); ++ii_s_tile)
      for (int ji_s_tile = ((127 + -1 * lid(1) + -16 * ii_s_tile + -16 * gid(2) + lid(0) >= 0 && 17 + -1 * lid(1) + -16 * ii_s_tile + lid(0) >= 0 && -2 + lid(1) + 16 * ii_s_tile + lid(0) >= 0 && -4 + lid(1) + 16 * ii_s_tile + 16 * gid(2) + lid(0) >= 0 && 125 + -1 * lid(1) + -16 * ii_s_tile + -16 * gid(2) + lid(0) + 16 * gid(1) >= 0 && 15 + -1 * lid(1) + -16 * ii_s_tile + lid(0) + 16 * gid(1) >= 0 && -4 + lid(1) + 16 * ii_s_tile + lid(0) + 16 * gid(1) >= 0 && -6 + lid(1) + 16 * ii_s_tile + 16 * gid(2) + lid(0) + 16 * gid(1) >= 0 && -26 + 13 * lid(0) + 240 * gid(1) >= 0) ? 0 : 1); ji_s_tile <= ((-7 + gid(1) == 0 && -7 + gid(2) == 0 && -15 + lid(1) + 16 * ii_s_tile >= 0 && 17 + -1 * lid(1) + -16 * ii_s_tile >= 0 && 48 + -1 * lid(1) + -32 * ii_s_tile + -1 * lid(0) + 16 * loopy_floor_div_pos_b_int32(-1 + -1 * lid(1) + lid(0), 16) >= 0 && 16 + lid(1) + -1 * lid(0) + 16 * loopy_floor_div_pos_b_int32(-1 + -1 * lid(1) + lid(0), 16) >= 0 && -1 + -1 * lid(1) + lid(0) + -16 * loopy_floor_div_pos_b_int32(-1 + -1 * lid(1) + lid(0), 16) >= 0) ? 2 + -1 * lid(1) + -1 * ii_s_tile + -1 * lid(0) + (15 * lid(1) + 15 * lid(0)) / 16 : (((ii_s_tile == 0 && 1 + -1 * lid(1) >= 0 && -26 + 13 * lid(1) + 240 * gid(2) >= 0 && -1 + gid(1) >= 0 && 6 + -1 * gid(1) >= 0) || (gid(1) == 0 && ii_s_tile == 0 && 1 + -1 * lid(1) >= 0 && -26 + 13 * lid(1) + 240 * gid(2) >= 0 && 17 + lid(1) + -1 * lid(0) + 16 * loopy_floor_div_pos_b_int32(-4 + lid(1) + lid(0), 16) >= 0 && -4 + lid(1) + lid(0) + -16 * loopy_floor_div_pos_b_int32(-4 + lid(1) + lid(0), 16) >= 0)) ? 1 + -1 * lid(0) + (1 + lid(1) + 15 * lid(0)) / 16 : (((-7 + gid(1) == 0 && -1 + ii_s_tile == 0 && -1 + lid(1) >= 0 && 6 + -1 * gid(2) >= 0 && 18 + -1 * lid(1) + -1 * lid(0) + 16 * loopy_floor_div_pos_b_int32(1 + -1 * lid(1) + lid(0), 16) >= 0 && 14 + lid(1) + -1 * lid(0) + 16 * loopy_floor_div_pos_b_int32(1 + -1 * lid(1) + lid(0), 16) >= 0 && 1 + -1 * lid(1) + lid(0) + -16 * loopy_floor_div_pos_b_int32(1 + -1 * lid(1) + lid(0), 16) >= 0) || (-7 + gid(2) == 0 && -1 + ii_s_tile == 0 && 1 + -1 * lid(1) >= 0 && gid(1) >= 0 && 6 + -1 * gid(1) >= 0 && 18 + -1 * lid(1) + -1 * lid(0) + 2 * gid(1) + 16 * loopy_floor_div_pos_b_int32(-3 + -1 * lid(1) + lid(0), 16) >= 0 && 18 + lid(1) + -1 * lid(0) + 16 * loopy_floor_div_pos_b_int32(-3 + -1 * lid(1) + lid(0), 16) >= 0 && -3 + -1 * lid(1) + lid(0) + -16 * loopy_floor_div_pos_b_int32(-3 + -1 * lid(1) + lid(0), 16) >= 0)) ? 1 + -1 * lid(1) + -1 * lid(0) + (2 + 15 * lid(1) + 15 * lid(0)) / 16 : ((-2 + lid(1) + 16 * ii_s_tile >= 0 && 17 + -1 * lid(1) + -16 * ii_s_tile >= 0 && 7 + -1 * ii_s_tile + -1 * gid(2) >= 0 && -4 + lid(1) + 16 * ii_s_tile + 16 * gid(2) >= 0 && 6 + -1 * gid(1) >= 0) ? 1 + -1 * lid(0) + (3 + 15 * lid(0)) / 16 : (((-1 + ii_s_tile == 0 && -2 + lid(1) >= 0 && 6 + -1 * gid(2) >= 0 && -1 + gid(1) >= 0 && 6 + -1 * gid(1) >= 0) || (gid(1) == 0 && -1 + ii_s_tile == 0 && -2 + lid(1) >= 0 && 6 + -1 * gid(2) >= 0 && 20 + -1 * lid(1) + -1 * lid(0) + 16 * loopy_floor_div_pos_b_int32(-1 + -1 * lid(1) + lid(0), 16) >= 0 && -1 + -1 * lid(1) + lid(0) + -16 * loopy_floor_div_pos_b_int32(-1 + -1 * lid(1) + lid(0), 16) >= 0)) ? 1 + -1 * lid(1) + -1 * lid(0) + (4 + 15 * lid(1) + 15 * lid(0)) / 16 : ((-7 + gid(1) == 0 && gid(2) == 0 && ii_s_tile == 0 && -2 + lid(1) >= 0 && 3 + -1 * lid(1) >= 0 && 12 + lid(1) + lid(0) + 16 * loopy_floor_div_pos_b_int32(-3 + lid(1) + -1 * lid(0), 16) >= 0 && -3 + lid(1) + -1 * lid(0) + -16 * loopy_floor_div_pos_b_int32(-3 + lid(1) + -1 * lid(0), 16) >= 0) ? -1 * lid(0) + (13 + lid(1) + 15 * lid(0)) / 16 : (((-7 + gid(1) == 0 && ii_s_tile == 0 && 2 + -1 * lid(1) >= 0 && -1 + gid(2) >= 0 && 17 + -1 * lid(1) + -1 * lid(0) + 16 * loopy_floor_div_pos_b_int32(-2 + lid(1) + lid(0), 16) >= 0 && 15 + lid(1) + -1 * lid(0) + 16 * loopy_floor_div_pos_b_int32(-2 + lid(1) + lid(0), 16) >= 0 && -2 + lid(1) + lid(0) + -16 * loopy_floor_div_pos_b_int32(-2 + lid(1) + lid(0), 16) >= 0) || (gid(2) == 0 && ii_s_tile == 0 && -2 + lid(1) >= 0 && 3 + -1 * lid(1) >= 0 && gid(1) >= 0 && 6 + -1 * gid(1) >= 0 && 21 + -1 * lid(1) + -1 * lid(0) + 16 * loopy_floor_div_pos_b_int32(-6 + lid(1) + lid(0), 16) >= 0 && 15 + lid(1) + -1 * lid(0) + 2 * gid(1) + 16 * loopy_floor_div_pos_b_int32(-6 + lid(1) + lid(0), 16) >= 0 && -6 + lid(1) + lid(0) + -16 * loopy_floor_div_pos_b_int32(-6 + lid(1) + lid(0), 16) >= 0)) ? -1 * lid(0) + (15 + lid(1) + 15 * lid(0)) / 16 : -5 + -1 * lid(0) + (97 + 15 * lid(0)) / 16))))))); ++ji_s_tile)
        u_ij_plane[20 * (16 * ii_s_tile + lid(1)) + 16 * ji_s_tile + lid(0)] = u[16384 * (-2 + 16 * gid(2) + 16 * ii_s_tile + lid(1)) + 128 * (-2 + 16 * gid(1) + 16 * ji_s_tile + lid(0)) + ki + 32 * gid(0)];
    if (-2 + lid(0) + 16 * gid(1) >= 0 && 125 + -1 * lid(0) + -16 * gid(1) >= 0 && -2 + lid(1) + 16 * gid(2) >= 0 && 125 + -1 * lid(1) + -16 * gid(2) >= 0)
    {
      if (-1 + ki >= 0 && -3 + ki + 32 * gid(0) >= 0 && 3 + -1 * 0 >= 0)
        u_k_buf[0] = u_k_buf[1];
      if (-4 + 0 == 0)
        u_k_buf[0] = u[16384 * (lid(1) + 16 * gid(2)) + 128 * (lid(0) + 16 * gid(1)) + -2 + ki + 32 * gid(0)];
      if (3 + -1 * 0 >= 0)
      {
        if (-2 + ki == 0 && gid(0) == 0)
          u_k_buf[0] = u[16384 * (lid(1) + 16 * gid(2)) + 128 * (lid(0) + 16 * gid(1)) + -2 + ki + 32 * gid(0)];
        if (ki == 0)
          u_k_buf[0] = u[16384 * (lid(1) + 16 * gid(2)) + 128 * (lid(0) + 16 * gid(1)) + -2 + ki + 32 * gid(0)];
      }
      if (-1 + ki >= 0 && -3 + ki + 32 * gid(0) >= 0 && 3 + -1 * 1 >= 0)
        u_k_buf[1] = u_k_buf[2];
      if (-4 + 1 == 0)
        u_k_buf[1] = u[16384 * (lid(1) + 16 * gid(2)) + 128 * (lid(0) + 16 * gid(1)) + -2 + 1 + ki + 32 * gid(0)];
      if (3 + -1 * 1 >= 0)
      {
        if (-2 + ki == 0 && gid(0) == 0)
          u_k_buf[1] = u[16384 * (lid(1) + 16 * gid(2)) + 128 * (lid(0) + 16 * gid(1)) + -2 + 1 + ki + 32 * gid(0)];
        if (ki == 0)
          u_k_buf[1] = u[16384 * (lid(1) + 16 * gid(2)) + 128 * (lid(0) + 16 * gid(1)) + -2 + 1 + ki + 32 * gid(0)];
      }
      if (-1 + ki >= 0 && -3 + ki + 32 * gid(0) >= 0 && 3 + -1 * 2 >= 0)
        u_k_buf[2] = u_k_buf[3];
      if (-4 + 2 == 0)
        u_k_buf[2] = u[16384 * (lid(1) + 16 * gid(2)) + 128 * (lid(0) + 16 * gid(1)) + -2 + 2 + ki + 32 * gid(0)];
      if (3 + -1 * 2 >= 0)
      {
        if (-2 + ki == 0 && gid(0) == 0)
          u_k_buf[2] = u[16384 * (lid(1) + 16 * gid(2)) + 128 * (lid(0) + 16 * gid(1)) + -2 + 2 + ki + 32 * gid(0)];
        if (ki == 0)
          u_k_buf[2] = u[16384 * (lid(1) + 16 * gid(2)) + 128 * (lid(0) + 16 * gid(1)) + -2 + 2 + ki + 32 * gid(0)];
      }
      if (-1 + ki >= 0 && -3 + ki + 32 * gid(0) >= 0 && 3 + -1 * 3 >= 0)
        u_k_buf[3] = u_k_buf[4];
      if (-4 + 3 == 0)
        u_k_buf[3] = u[16384 * (lid(1) + 16 * gid(2)) + 128 * (lid(0) + 16 * gid(1)) + -2 + 3 + ki + 32 * gid(0)];
      if (3 + -1 * 3 >= 0)
      {
        if (-2 + ki == 0 && gid(0) == 0)
          u_k_buf[3] = u[16384 * (lid(1) + 16 * gid(2)) + 128 * (lid(0) + 16 * gid(1)) + -2 + 3 + ki + 32 * gid(0)];
        if (ki == 0)
          u_k_buf[3] = u[16384 * (lid(1) + 16 * gid(2)) + 128 * (lid(0) + 16 * gid(1)) + -2 + 3 + ki + 32 * gid(0)];
      }
      if (-1 + ki >= 0 && -3 + ki + 32 * gid(0) >= 0 && 3 + -1 * 4 >= 0)
        u_k_buf[4] = u_k_buf[5];
      if (-4 + 4 == 0)
        u_k_buf[4] = u[16384 * (lid(1) + 16 * gid(2)) + 128 * (lid(0) + 16 * gid(1)) + -2 + 4 + ki + 32 * gid(0)];
      if (3 + -1 * 4 >= 0)
      {
        if (-2 + ki == 0 && gid(0) == 0)
          u_k_buf[4] = u[16384 * (lid(1) + 16 * gid(2)) + 128 * (lid(0) + 16 * gid(1)) + -2 + 4 + ki + 32 * gid(0)];
        if (ki == 0)
          u_k_buf[4] = u[16384 * (lid(1) + 16 * gid(2)) + 128 * (lid(0) + 16 * gid(1)) + -2 + 4 + ki + 32 * gid(0)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE) /* for u_ij_plane (insn_l_update depends on u_plane_compute) */;
    if (-2 + lid(0) + 16 * gid(1) >= 0 && 125 + -1 * lid(0) + -16 * gid(1) >= 0 && -2 + lid(1) + 16 * gid(2) >= 0 && 125 + -1 * lid(1) + -16 * gid(2) >= 0)
    {
      for (int l = -2; l <= 2; ++l)
        acc_l = acc_l + c[2 + l] * (u_ij_plane[20 * (2 + lid(1) + -1 * l) + 2 + lid(0)] + u_ij_plane[20 * (2 + lid(1)) + 2 + lid(0) + -1 * l] + u_k_buf[2 + -1 * l]);
      lap_u[16384 * (lid(1) + 16 * gid(2)) + 128 * (lid(0) + 16 * gid(1)) + ki + 32 * gid(0)] = acc_l;
    }
  }
}