#include <stdio.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#define CUDA_CHECK(call) \
    do { \
        CUresult error = call; \
        if (error != CUDA_SUCCESS) { \
            const char* errorStr; \
            cuGetErrorString(error, &errorStr); \
            fprintf(stderr, "CUDA Driver API error at %s:%d - %s (code: %d)\n", __FILE__, __LINE__, errorStr, error); \
            exit(1); \
        } \
    } while(0)

#define CUDA_RT_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Runtime API error at %s:%d - %s (code: %d)\n", __FILE__, __LINE__, cudaGetErrorString(error), error); \
            exit(1); \
        } \
    } while(0)


__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  nvtxRangePush("Initialization");
  // Initialize CUDA driver API
  CUDA_CHECK(cuInit(0));
  printf("CUDA Driver API initialized successfully\n");
  
  int N = 1<<30;
  printf("Allocating %d elements (%zu MB)\n", N, (size_t)N * sizeof(float) / (1024*1024));
  
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));
  
  if (!x || !y) {
    fprintf(stderr, "Failed to allocate host memory\n");
    exit(1);
  }
  printf("Host memory allocated successfully\n");

  size_t size = N * sizeof(float);
  size_t granularity;
  
  nvtxRangePush("VMM Setup");
  printf("Setting up VMM for GPU 0...\n");
  
  // Check and enable peer-to-peer access between GPUs
  int canAccessPeer;
  CUDA_RT_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1));
  if (canAccessPeer) {
    CUDA_RT_CHECK(cudaSetDevice(0));
    CUDA_RT_CHECK(cudaDeviceEnablePeerAccess(1, 0));
    printf("Peer-to-peer access enabled from GPU 0 to GPU 1\n");
  } else {
    printf("Warning: Peer-to-peer access not available between GPU 0 and GPU 1\n");
  }
  
  // Get memory allocation properties
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0; // GPU 0
  
  CUDA_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  printf("Memory granularity: %zu bytes\n", granularity);
  
  // Round up to granularity
  size_t padded_size = ((size + granularity - 1) / granularity) * granularity;
  printf("Padded size: %zu bytes (%zu MB)\n", padded_size, padded_size / (1024*1024));
  
  // Allocate virtual address range
  CUdeviceptr d_x_ptr, d_y_ptr;
  CUDA_CHECK(cuMemAddressReserve(&d_x_ptr, padded_size, 0, 0, 0));
  CUDA_CHECK(cuMemAddressReserve(&d_y_ptr, padded_size, 0, 0, 0));
  printf("Virtual address ranges reserved: d_x=0x%lx, d_y=0x%lx\n", d_x_ptr, d_y_ptr);
  
  // Create physical memory handle
  CUmemGenericAllocationHandle handle_x, handle_y;
  CUDA_CHECK(cuMemCreate(&handle_x, padded_size, &prop, 0));
  CUDA_CHECK(cuMemCreate(&handle_y, padded_size, &prop, 0));
  printf("Physical memory handles created\n");
  
  // Map virtual addresses to physical memory
  CUDA_CHECK(cuMemMap(d_x_ptr, padded_size, 0, handle_x, 0));
  CUDA_CHECK(cuMemMap(d_y_ptr, padded_size, 0, handle_y, 0));
  printf("Virtual addresses mapped to physical memory\n");
  
  // Set access for both GPUs initially (for warmup phase)
  CUmemAccessDesc accessDesc[2];
  accessDesc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc[0].location.id = 0;
  accessDesc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  
  accessDesc[1].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc[1].location.id = 1;
  accessDesc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  
  CUDA_CHECK(cuMemSetAccess(d_x_ptr, padded_size, accessDesc, 2));
  CUDA_CHECK(cuMemSetAccess(d_y_ptr, padded_size, accessDesc, 2));
  printf("Access permissions set for both GPUs (for warmup)\n");
  nvtxRangePop();
  
  d_x = (float*)d_x_ptr;
  d_y = (float*)d_y_ptr;

  printf("Initializing host data...\n");
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  printf("Host data initialized\n");

  printf("Copying data from host to device...\n");
  CUDA_RT_CHECK(cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_RT_CHECK(cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice));
  printf("Data copied to device successfully\n");

  nvtxRangePush("Warmup");
  printf("Starting warmup kernels...\n");
  
  CUDA_RT_CHECK(cudaSetDevice(0));
  printf("Running warmup kernel on GPU 0\n");
  saxpy<<<(N+255)/256, 256>>>(N, 1.0f, d_x, d_y);
  CUDA_RT_CHECK(cudaDeviceSynchronize());
  printf("GPU 0 warmup completed\n");
  
  CUDA_RT_CHECK(cudaSetDevice(1));
  printf("Running warmup kernel on GPU 1\n");
  saxpy<<<(N+255)/256, 256>>>(N, 1.0f, d_x, d_y);
  CUDA_RT_CHECK(cudaDeviceSynchronize());
  printf("GPU 1 warmup completed\n");
  
  printf("Warmup done\n");
  nvtxRangePop();
  nvtxRangePop(); // End initialization


  // experiment 1: kernel0 on gpu 0, kernel 1 on gpu 1, data dependency via VMM
  nvtxRangePush("Kernel on GPU 0 + Memory Prep");
  CUDA_RT_CHECK(cudaSetDevice(0));
  int device0;
  CUDA_RT_CHECK(cudaGetDevice(&device0));
  printf("Running first kernel on GPU %d\n", device0);
  
  cudaStream_t stream0;
  CUDA_RT_CHECK(cudaStreamCreate(&stream0));
  cudaEvent_t done0; 
  CUDA_RT_CHECK(cudaEventCreateWithFlags(&done0, cudaEventDisableTiming));

  // Start kernel on GPU 0
  printf("Launching kernel on GPU %d\n", device0);
  saxpy<<<(N+255)/256, 256, 0, stream0>>>(N, 2.0f, d_x, d_y);
  CUDA_RT_CHECK(cudaEventRecord(done0, stream0));
  
  // While GPU 0 kernel is running, prepare GPU 1 memory in parallel
  nvtxRangePush("GPU 1 Memory Preparation");
  printf("Preparing GPU 1 memory while GPU 0 kernel runs...\n");
  
  // Create new physical memory on GPU 1 (this can run in parallel with GPU 0 kernel)
  CUmemAllocationProp prop_gpu1 = {};
  prop_gpu1.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop_gpu1.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop_gpu1.location.id = 1; // GPU 1

  CUmemGenericAllocationHandle handle_x_gpu1, handle_y_gpu1;
  CUDA_CHECK(cuMemCreate(&handle_x_gpu1, padded_size, &prop_gpu1, 0));
  CUDA_CHECK(cuMemCreate(&handle_y_gpu1, padded_size, &prop_gpu1, 0));
  printf("GPU 1 physical memory handles created\n");
  
  // Allocate temporary virtual address ranges for GPU 1 memory
  CUdeviceptr temp_x_ptr, temp_y_ptr;
  CUDA_CHECK(cuMemAddressReserve(&temp_x_ptr, padded_size, 0, 0, 0));
  CUDA_CHECK(cuMemAddressReserve(&temp_y_ptr, padded_size, 0, 0, 0));
  printf("Temporary virtual address ranges reserved: temp_x=0x%lx, temp_y=0x%lx\n", temp_x_ptr, temp_y_ptr);

  // Map temporary virtual addresses to GPU 1's physical memory
  CUDA_CHECK(cuMemMap(temp_x_ptr, padded_size, 0, handle_x_gpu1, 0));
  CUDA_CHECK(cuMemMap(temp_y_ptr, padded_size, 0, handle_y_gpu1, 0));
  printf("Temporary virtual addresses mapped to GPU 1 physical memory\n");

  // Set access for GPU 1 on temporary memory
  CUmemAccessDesc temp_accessDesc[1];
  temp_accessDesc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  temp_accessDesc[0].location.id = 1;
  temp_accessDesc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  CUDA_CHECK(cuMemSetAccess(temp_x_ptr, padded_size, temp_accessDesc, 1));
  CUDA_CHECK(cuMemSetAccess(temp_y_ptr, padded_size, temp_accessDesc, 1));
  printf("Access permissions set for GPU 1 temporary memory\n");
  
  printf("GPU 1 memory preparation complete\n");
  nvtxRangePop();
  
  // Now wait for GPU 0 kernel to complete before copying data
  printf("Waiting for GPU 0 kernel to complete...\n");
  CUDA_RT_CHECK(cudaStreamWaitEvent(0, done0, 0));  // Wait for kernel completion
  printf("GPU %d kernel completed\n", device0);
  
  // Restrict access to GPU 0 only before remapping
  printf("Restricting memory access to GPU 0 only before remapping...\n");
  CUmemAccessDesc gpu0_only_access[1];
  gpu0_only_access[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  gpu0_only_access[0].location.id = 0;
  gpu0_only_access[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  
  CUDA_CHECK(cuMemSetAccess(d_x_ptr, padded_size, gpu0_only_access, 1));
  CUDA_CHECK(cuMemSetAccess(d_y_ptr, padded_size, gpu0_only_access, 1));
  printf("Memory access restricted to GPU 0 only\n");
  
  nvtxRangePop();

  nvtxRangePush("Data Migration");
  printf("Starting data migration from GPU 0 to GPU 1...\n");

  // Copy data directly from GPU 0 to GPU 1 using CUDA Driver API
  printf("Copying data from GPU 0 to GPU 1...\n");
  
  // Try direct device-to-device copy using Driver API
  CUresult result_x = cuMemcpyDtoD(temp_x_ptr, d_x_ptr, size);
  CUresult result_y = cuMemcpyDtoD(temp_y_ptr, d_y_ptr, size);
  
  if (result_x != CUDA_SUCCESS || result_y != CUDA_SUCCESS) {
    printf("Direct D2D copy failed, trying cudaMemcpyPeer...\n");
    
    // Alternative: Use cudaMemcpyPeer for cross-GPU copy
    CUDA_RT_CHECK(cudaSetDevice(0));
    CUDA_RT_CHECK(cudaMemcpyPeer((float*)temp_x_ptr, 1, d_x, 0, size));
    CUDA_RT_CHECK(cudaMemcpyPeer((float*)temp_y_ptr, 1, d_y, 0, size));
    
    // Verify the copy worked
    CUDA_RT_CHECK(cudaSetDevice(1));
    float test_copy_x, test_copy_y;
    CUDA_RT_CHECK(cudaMemcpy(&test_copy_x, (float*)temp_x_ptr, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_RT_CHECK(cudaMemcpy(&test_copy_y, (float*)temp_y_ptr, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Copy verification: temp_x[0]=%f, temp_y[0]=%f\n", test_copy_x, test_copy_y);
  } else {
    // Synchronize to ensure copy is complete
    CUDA_CHECK(cuCtxSynchronize());
    printf("Direct D2D copy successful\n");
  }
  
  printf("Data copy completed\n");

  // Verify that the data was actually copied before proceeding
  printf("Verifying data copy before remapping...\n");
  CUDA_RT_CHECK(cudaSetDevice(1));
  float verify_copy_x, verify_copy_y;
  CUDA_RT_CHECK(cudaMemcpy(&verify_copy_x, (float*)temp_x_ptr, sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_RT_CHECK(cudaMemcpy(&verify_copy_y, (float*)temp_y_ptr, sizeof(float), cudaMemcpyDeviceToHost));
  printf("Pre-remap verification: temp_x[0]=%f, temp_y[0]=%f\n", verify_copy_x, verify_copy_y);
  
  if (verify_copy_x == 0.0f && verify_copy_y == 0.0f) {
    fprintf(stderr, "ERROR: Data copy verification failed - data not properly copied to GPU 1\n");
    exit(1);
  }

  // Now unmap the original mappings
  printf("Unmapping original virtual addresses...\n");
  CUDA_CHECK(cuMemUnmap(d_x_ptr, padded_size));
  CUDA_CHECK(cuMemUnmap(d_y_ptr, padded_size));
  CUDA_CHECK(cuMemRelease(handle_x));
  CUDA_CHECK(cuMemRelease(handle_y));
  printf("Original mappings released\n");

  // Remap original virtual addresses to GPU 1's physical memory
  printf("Remapping virtual addresses to GPU 1 physical memory...\n");
  CUDA_CHECK(cuMemMap(d_x_ptr, padded_size, 0, handle_x_gpu1, 0));
  CUDA_CHECK(cuMemMap(d_y_ptr, padded_size, 0, handle_y_gpu1, 0));

  // Set access for GPU 1 on remapped memory
  CUmemAccessDesc accessDesc_remap[1];
  accessDesc_remap[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc_remap[0].location.id = 1;
  accessDesc_remap[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  CUDA_CHECK(cuMemSetAccess(d_x_ptr, padded_size, accessDesc_remap, 1));
  CUDA_CHECK(cuMemSetAccess(d_y_ptr, padded_size, accessDesc_remap, 1));
  printf("Access permissions set for remapped memory on GPU 1\n");
  
  // Verify the remapping worked by checking memory access
  printf("Verifying VMM remapping...\n");
  CUDA_RT_CHECK(cudaSetDevice(1));
  
  // Test that we can read from the remapped memory
  float verify_value;
  CUDA_RT_CHECK(cudaMemcpy(&verify_value, d_x, sizeof(float), cudaMemcpyDeviceToHost));
  printf("VMM remapping verification successful, first value: %f\n", verify_value);

  // No need to copy data again - we just need to clean up temporary virtual address ranges
  // The data is already in the physical memory that we've remapped to d_x and d_y
  printf("Cleaning up temporary virtual address ranges...\n");
  CUDA_CHECK(cuMemUnmap(temp_x_ptr, padded_size));
  CUDA_CHECK(cuMemUnmap(temp_y_ptr, padded_size));
  CUDA_CHECK(cuMemAddressFree(temp_x_ptr, padded_size));
  CUDA_CHECK(cuMemAddressFree(temp_y_ptr, padded_size));

  printf("Remapping complete. Physical pages now on GPU 1\n");
  nvtxRangePop();

  // launch another kernel on gpu 1
  nvtxRangePush("Kernel on GPU 1");
  CUDA_RT_CHECK(cudaSetDevice(1));
  int device;
  CUDA_RT_CHECK(cudaGetDevice(&device));
  printf("Running kernels on GPU %d\n", device);
  
  // Verify memory access before launching kernels
  printf("Verifying memory access on GPU %d...\n", device);
  
  // Test memory access with a simple read
  float test_value;
  CUDA_RT_CHECK(cudaMemcpy(&test_value, d_x, sizeof(float), cudaMemcpyDeviceToHost));
  printf("Memory access test successful, first value: %f\n", test_value);
  
  cudaStream_t stream1;
  CUDA_RT_CHECK(cudaStreamCreate(&stream1));
  
  // Launch kernels one at a time with error checking
  for (int i = 0; i < 2; i++){
    printf("Launching kernel %d on GPU %d\n", i, device);
    saxpy<<<(N+255)/256, 256, 0, stream1>>>(N, 1.0f, d_x, d_y);
    
    // Check for errors after each kernel launch
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(error));
      exit(1);
    }
    
    // Synchronize after each kernel to catch errors early
    CUDA_RT_CHECK(cudaStreamSynchronize(stream1));
    printf("Kernel %d completed successfully\n", i);
  }
  
  printf("GPU %d kernels completed\n", device);
  nvtxRangePop();

  printf("Copying final results back to host...\n");
  CUDA_RT_CHECK(cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost));
  printf("Results copied to host\n");

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-8.0f));
  printf("Max error: %f\n", maxError);

  // Cleanup VMM resources (now handle_x_gpu1, handle_y_gpu1)
  printf("Cleaning up VMM resources...\n");
  CUDA_CHECK(cuMemUnmap(d_x_ptr, padded_size));
  CUDA_CHECK(cuMemUnmap(d_y_ptr, padded_size));
  CUDA_CHECK(cuMemRelease(handle_x_gpu1));
  CUDA_CHECK(cuMemRelease(handle_y_gpu1));
  CUDA_CHECK(cuMemAddressFree(d_x_ptr, padded_size));
  CUDA_CHECK(cuMemAddressFree(d_y_ptr, padded_size));
  printf("VMM resources cleaned up\n");
  
  free(x);
  free(y);
  printf("Host memory freed\n");
  printf("Program completed successfully!\n");
}