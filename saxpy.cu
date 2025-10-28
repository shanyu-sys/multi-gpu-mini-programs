#include <stdio.h>
#include <unistd.h>
#include <cuda.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  float *x, *y, *d_x, *d_y, *d_x_managed, *d_y_managed;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMallocManaged(&d_x_managed, N*sizeof(float)); 
  cudaMallocManaged(&d_y_managed, N*sizeof(float));
  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 0.0f;
  }

  cudaMemcpy(d_x_managed, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y_managed, y, N*sizeof(float), cudaMemcpyHostToDevice);
  
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

  printf("benchmark peer access\n");
  for (int i = 0; i < 3; i++) {
    cudaSetDevice(0);
    // cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
    // Perform SAXPY on 1M elements
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float milliseconds = 0;
    for (int j = 0; j < 3; j++) {
      cudaEventRecord(start);
      saxpy<<<(N+255)/256, 256>>>(N, 1.0f, d_x, d_y);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaDeviceSynchronize();
      cudaEventElapsedTime(&milliseconds, start, stop);
      printf("Device 0 kernel time: %f ms\n", milliseconds);
    }

    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0);
    for (int j = 0; j < 3; j++) {
      cudaEventRecord(start);
      saxpy<<<(N+255)/256, 256>>>(N, 1.0f, d_x, d_y);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaDeviceSynchronize();
      cudaEventElapsedTime(&milliseconds, start, stop);
      printf("Device 1 kernel time: %f ms\n", milliseconds);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    float max_value = 0.0f;
    for (int i = 0; i < N; i++) {
      if (y[i] > max_value) max_value = y[i];
    }
    printf("Max value: %f\n", max_value);
    cudaDeviceDisablePeerAccess(0);
  }

  printf("benchmark managed\n");
  for (int i = 0; i < 3; i++) {
    cudaSetDevice(0);
    // cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
    // Perform SAXPY on 1M elements
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float milliseconds = 0;
    for (int j = 0; j < 3; j++) {
      cudaEventRecord(start);
      saxpy<<<(N+255)/256, 256>>>(N, 1.0f, d_x_managed, d_y_managed);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      printf("Device 0 kernel time: %f ms\n", milliseconds);
    }

    cudaSetDevice(1);
    // cudaDeviceEnablePeerAccess(0, 0);
    for (int j = 0; j < 3; j++) {
      cudaEventRecord(start);
      saxpy<<<(N+255)/256, 256>>>(N, 1.0f, d_x_managed, d_y_managed);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      printf("Device 1 kernel time: %f ms\n", milliseconds);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(y, d_y_managed, N*sizeof(float), cudaMemcpyDeviceToHost);

    float max_value = 0.0f;
    for (int i = 0; i < N; i++) {
      if (y[i] > max_value) max_value = y[i];
    }
    printf("Max value: %f\n", max_value);
  }

  printf("benchmark managed with prefetch\n");
  for (int i = 0; i < 3; i++) {
    cudaSetDevice(0);
    // cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
    // Perform SAXPY on 1M elements
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float milliseconds = 0;
    for (int j = 0; j < 3; j++) {
      cudaEventRecord(start);
      saxpy<<<(N+255)/256, 256>>>(N, 1.0f, d_x_managed, d_y_managed);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      printf("Device 0 kernel time: %f ms\n", milliseconds);
    }

    cudaSetDevice(1);
    // cudaDeviceEnablePeerAccess(0, 0);
    cudaMemPrefetchAsync(d_x_managed, N*sizeof(float), 1);
    cudaMemPrefetchAsync(d_y_managed, N*sizeof(float), 1);
    for (int j = 0; j < 3; j++) {
      cudaEventRecord(start);
      saxpy<<<(N+255)/256, 256>>>(N, 1.0f, d_x_managed, d_y_managed);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      printf("Device 1 kernel time: %f ms\n", milliseconds);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(y, d_y_managed, N*sizeof(float), cudaMemcpyDeviceToHost);

    float max_value = 0.0f;
    for (int i = 0; i < N; i++) {
      if (y[i] > max_value) max_value = y[i];
    }
    printf("Max value: %f\n", max_value);
  }

  // sleep 5 seconds
  usleep(5000000);

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}