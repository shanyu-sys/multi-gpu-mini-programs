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
  // Initialize CUDA driver API
  cuInit(0);
  int N = 1<<30;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  size_t size = N * sizeof(float);
  size_t granularity;
  
  // Get memory allocation properties
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0; // GPU 0
  
  cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  
  // Round up to granularity
  size_t padded_size = ((size + granularity - 1) / granularity) * granularity;
  
  // Allocate virtual address range
  CUdeviceptr d_x_ptr, d_y_ptr;
  cuMemAddressReserve(&d_x_ptr, padded_size, 0, 0, 0);
  cuMemAddressReserve(&d_y_ptr, padded_size, 0, 0, 0);
  
  // Create physical memory handle
  CUmemGenericAllocationHandle handle_x, handle_y;
  cuMemCreate(&handle_x, padded_size, &prop, 0);
  cuMemCreate(&handle_y, padded_size, &prop, 0);
  
  // Map virtual addresses to physical memory
  cuMemMap(d_x_ptr, padded_size, 0, handle_x, 0);
  cuMemMap(d_y_ptr, padded_size, 0, handle_y, 0);
  
  // Set access for both GPUs
  CUmemAccessDesc accessDesc[2];
  accessDesc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc[0].location.id = 0;
  accessDesc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  
  accessDesc[1].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc[1].location.id = 1;
  accessDesc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  
  cuMemSetAccess(d_x_ptr, padded_size, accessDesc, 2);
  cuMemSetAccess(d_y_ptr, padded_size, accessDesc, 2);
  
  d_x = (float*)d_x_ptr;
  d_y = (float*)d_y_ptr;

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  printf("Warmup \n");
  cudaSetDevice(0);
  saxpy<<<(N+255)/256, 256>>>(N, 1.0f, d_x, d_y);
  cudaDeviceSynchronize();
  cudaSetDevice(1);
  saxpy<<<(N+255)/256, 256>>>(N, 1.0f, d_x, d_y);
  cudaDeviceSynchronize();
  printf("Warmup done\n");


  // experiment 1: kernel0 on gpu 0, kernel 1 on gpu 1, data dependency via VMM
  cudaSetDevice(0);
  cudaStream_t stream0;
  cudaStreamCreate(&stream0);
  cudaEvent_t done0; 
  cudaEventCreateWithFlags(&done0, cudaEventDisableTiming);

  // Perform SAXPY on 1M elements
  saxpy<<<(N+255)/256, 256, 0, stream0>>>(N, 2.0f, d_x, d_y);
  cudaEventRecord(done0, stream0);

  // launch another kernel on gpu 1
  cudaSetDevice(1);
  cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  cudaStreamWaitEvent(stream1, done0, 0);
  for (int i = 0; i < 2; i++){
    saxpy<<<(N+255)/256, 256, 0, stream1>>>(N, 1.0f, d_x, d_y);
  }
  cudaDeviceSynchronize();

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-8.0f));
  printf("Max error: %f\n", maxError);

  // Cleanup VMM resources
  cuMemUnmap(d_x_ptr, padded_size);
  cuMemUnmap(d_y_ptr, padded_size);
  cuMemRelease(handle_x);
  cuMemRelease(handle_y);
  cuMemAddressFree(d_x_ptr, padded_size);
  cuMemAddressFree(d_y_ptr, padded_size);
  
  free(x);
  free(y);
}