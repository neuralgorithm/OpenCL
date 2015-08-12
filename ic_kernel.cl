#define N 1000
#define decay 0.99
#define I 1.0
#define Kappa 2.0

__kernel void ic_compute(const int t, __global const float *w, __global float *u, __global float *z, __global float *result)
{
  int i = get_global_id(0);
  float r = 0;
  for(int j = 0; j < N; j++){ r += w[j+N*i] * z[j]; }

  u[i] = decay*u[i] + (1 - decay)*I - Kappa*r/N;
  z[i] = fmax(u[i], 0);

  result[i+N*t] = z[i];
}
