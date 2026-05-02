#define bIdx(N) ((int) blockIdx.N)
#define tIdx(N) ((int) threadIdx.N)

extern "C" __global__ void __launch_bounds__(256) loopy_kernel(double const *__restrict__ u, double *__restrict__ lap_u, double const *__restrict__ c)
{
  double acc_l;

  if (253 + -16 * bIdx(y) + -1 * tIdx(x) >= 0 && -2 + 16 * bIdx(z) + tIdx(y) >= 0 && 253 + -16 * bIdx(z) + -1 * tIdx(y) >= 0 && -2 + 16 * bIdx(y) + tIdx(x) >= 0)
    for (int ki = ((-1 + bIdx(x) >= 0) ? 0 : 2); ki <= ((-7 + bIdx(x) == 0) ? 29 : 31); ++ki)
    {
      acc_l = (double) (0.0);
      for (int l = -2; l <= 2; ++l)
        acc_l = acc_l + c[2 + l] * (u[65536 * (-1 * l + 16 * bIdx(z) + tIdx(y)) + 256 * (16 * bIdx(y) + tIdx(x)) + 32 * bIdx(x) + ki] + u[65536 * (16 * bIdx(z) + tIdx(y)) + 256 * (-1 * l + 16 * bIdx(y) + tIdx(x)) + 32 * bIdx(x) + ki] + u[65536 * (16 * bIdx(z) + tIdx(y)) + 256 * (16 * bIdx(y) + tIdx(x)) + -1 * l + 32 * bIdx(x) + ki]);
      lap_u[65536 * (16 * bIdx(z) + tIdx(y)) + 256 * (16 * bIdx(y) + tIdx(x)) + 32 * bIdx(x) + ki] = acc_l;
    }
}