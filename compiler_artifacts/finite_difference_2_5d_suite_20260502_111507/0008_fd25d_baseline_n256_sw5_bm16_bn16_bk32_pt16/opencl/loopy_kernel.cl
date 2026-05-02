#define lid(N) ((int) get_local_id(N))
#define gid(N) ((int) get_group_id(N))
#if __OPENCL_C_VERSION__ < 120
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

__kernel void __attribute__ ((reqd_work_group_size(16, 16, 1))) loopy_kernel(__global double const *__restrict__ u, __global double *__restrict__ lap_u, __global double const *__restrict__ c)
{
  double acc_l;

  if (253 + -16 * gid(1) + -1 * lid(0) >= 0 && -2 + 16 * gid(2) + lid(1) >= 0 && 253 + -16 * gid(2) + -1 * lid(1) >= 0 && -2 + 16 * gid(1) + lid(0) >= 0)
    for (int ki = ((-1 + gid(0) >= 0) ? 0 : 2); ki <= ((-7 + gid(0) == 0) ? 29 : 31); ++ki)
    {
      acc_l = (double) (0.0);
      for (int l = -2; l <= 2; ++l)
        acc_l = acc_l + c[2 + l] * (u[65536 * (-1 * l + 16 * gid(2) + lid(1)) + 256 * (16 * gid(1) + lid(0)) + 32 * gid(0) + ki] + u[65536 * (16 * gid(2) + lid(1)) + 256 * (-1 * l + 16 * gid(1) + lid(0)) + 32 * gid(0) + ki] + u[65536 * (16 * gid(2) + lid(1)) + 256 * (16 * gid(1) + lid(0)) + -1 * l + 32 * gid(0) + ki]);
      lap_u[65536 * (16 * gid(2) + lid(1)) + 256 * (16 * gid(1) + lid(0)) + 32 * gid(0) + ki] = acc_l;
    }
}