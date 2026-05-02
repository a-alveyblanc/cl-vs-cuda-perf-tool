#define bIdx(N) ((int) blockIdx.N)
#define tIdx(N) ((int) threadIdx.N)

extern "C" __global__ void __launch_bounds__(256) loopy_kernel(double const *__restrict__ u, double *__restrict__ lap_u, double const *__restrict__ c)
{
  double acc_l;

  if (125 + -16 * bIdx(y) + -1 * tIdx(x) >= 0 && -2 + 16 * bIdx(z) + tIdx(y) >= 0 && 125 + -16 * bIdx(z) + -1 * tIdx(y) >= 0 && -2 + 16 * bIdx(y) + tIdx(x) >= 0)
    for (int ki = ((-1 + bIdx(x) >= 0) ? 0 : 2); ki <= ((-3 + bIdx(x) == 0) ? 29 : 31); ++ki)
    {
      acc_l = (double) (0.0);
      for (int l = -2; l <= 2; ++l)
        acc_l = acc_l + c[2 + l] * (u[16384 * (-1 * l + 16 * bIdx(z) + tIdx(y)) + 128 * (16 * bIdx(y) + tIdx(x)) + 32 * bIdx(x) + ki] + u[16384 * (16 * bIdx(z) + tIdx(y)) + 128 * (-1 * l + 16 * bIdx(y) + tIdx(x)) + 32 * bIdx(x) + ki] + u[16384 * (16 * bIdx(z) + tIdx(y)) + 128 * (16 * bIdx(y) + tIdx(x)) + -1 * l + 32 * bIdx(x) + ki]);
      lap_u[16384 * (16 * bIdx(z) + tIdx(y)) + 128 * (16 * bIdx(y) + tIdx(x)) + 32 * bIdx(x) + ki] = acc_l;
    }
}