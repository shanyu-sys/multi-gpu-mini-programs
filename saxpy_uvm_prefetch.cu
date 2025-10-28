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
  float *x, *y;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));


  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  cudaSetDevice(0);
  cudaStream_t stream0;
  cudaStreamCreate(&stream0);
  cudaEvent_t done0; 
  cudaEventCreateWithFlags(&done0, cudaEventDisableTiming);

  for (int i = 0; i < 2; i++){
    saxpy<<<numBlocks, blockSize, 0, stream0>>>(N, 2.0f, x, y);
  }
  cudaEventRecord(done0, stream0);

  cudaDeviceEnablePeerAccess(0, 0);   // Enable peer-to-peer access

  // launch another kernel on gpu 1
  cudaSetDevice(1);
  cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  cudaStreamWaitEvent(stream1, done0, 0);

  cudaMemPrefetchAsync(x, N*sizeof(float), 1);
  for (int i = 0; i < 2; i++){
    saxpy<<<(N+255)/256, 256, 0, stream1>>>(N, 1.0f, x, y);
  }
  cudaDeviceSynchronize();

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-8.0f));
  printf("Max error: %f\n", maxError);

  cudaFree(x);
  cudaFree(y); 
}