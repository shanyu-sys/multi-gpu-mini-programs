#include <stdio.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  int N = 1<<30;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  cudaDeviceEnablePeerAccess(1, 0);
  printf("Warmup \n");
  cudaSetDevice(0);
  saxpy<<<(N+255)/256, 256>>>(N, 1.0f, d_x, d_y);
  cudaDeviceSynchronize();
  cudaSetDevice(1);
  cudaDeviceEnablePeerAccess(0, 0);
  saxpy<<<(N+255)/256, 256>>>(N, 1.0f, d_x, d_y);
  cudaDeviceSynchronize();
  printf("Warmup done\n");


  // experiment 1: kernel0 on gpu 0, kernel 1 on gpu 1, data dependency via peer access
  cudaSetDevice(0);
  cudaStream_t stream0;
  cudaStreamCreate(&stream0);
  cudaEvent_t done0; 
  cudaEventCreateWithFlags(&done0, cudaEventDisableTiming);

  // Perform SAXPY on 1M elements
  saxpy<<<(N+255)/256, 256, 0, stream0>>>(N, 2.0f, d_x, d_y);
  cudaEventRecord(done0, stream0);

  cudaDeviceEnablePeerAccess(0, 0);   // Enable peer-to-peer access

  // launch another kernel on gpu 1
  cudaSetDevice(1);
  cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  cudaStreamWaitEvent(stream1, done0, 0);
  for (int i = 0; i < 2; i++){
    saxpy<<<(N+255)/256, 256, 0, stream1>>>(N, 1.0f, d_x, d_y);
  }

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-8.0f));
  printf("Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}