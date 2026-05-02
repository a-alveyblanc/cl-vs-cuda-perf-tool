#define bIdx(N) ((int) blockIdx.N)
#define tIdx(N) ((int) threadIdx.N)

extern "C" __global__ void __launch_bounds__(256) loopy_kernel(double const *__restrict__ u, double *__restrict__ lap_u, double const *__restrict__ c)
{
  double acc_l;

  if (59 + -16 * bIdx(y) + -1 * tIdx(x) >= 0 && -4 + 16 * bIdx(z) + tIdx(y) >= 0 && 59 + -16 * bIdx(z) + -1 * tIdx(y) >= 0 && -4 + 16 * bIdx(y) + tIdx(x) >= 0)
    for (int ki = ((-1 + bIdx(x) == 0) ? 0 : 4); ki <= ((-1 + bIdx(x) == 0) ? 27 : 31); ++ki)
    {
      acc_l = (double) (0.0);
      for (int l = -4; l <= 4; ++l)
        acc_l = acc_l + c[4 + l] * (u[4096 * (-1 * l + 16 * bIdx(z) + tIdx(y)) + 64 * (16 * bIdx(y) + tIdx(x)) + 32 * bIdx(x) + ki] + u[4096 * (16 * bIdx(z) + tIdx(y)) + 64 * (-1 * l + 16 * bIdx(y) + tIdx(x)) + 32 * bIdx(x) + ki] + u[4096 * (16 * bIdx(z) + tIdx(y)) + 64 * (16 * bIdx(y) + tIdx(x)) + -1 * l + 32 * bIdx(x) + ki]);
      lap_u[4096 * (16 * bIdx(z) + tIdx(y)) + 64 * (16 * bIdx(y) + tIdx(x)) + 32 * bIdx(x) + ki] = acc_l;
    }
}