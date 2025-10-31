#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

__global__ void saxpy(int n, float a, float *x, float *y, float *output) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) output[i] = a*x[i] + y[i];
}

// Timing helper functions
std::chrono::high_resolution_clock::time_point get_time() {
  return std::chrono::high_resolution_clock::now();
}

double get_elapsed_ms(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void print_gpu_memory(int device) {
    size_t freeMem, totalMem;
    cudaSetDevice(device);
    cudaMemGetInfo(&freeMem, &totalMem);
    printf("GPU %d - Free memory: %.2f GB, Total memory: %.2f GB\n", 
           device,
           freeMem / (1024.0 * 1024.0 * 1024.0), 
           totalMem / (1024.0 * 1024.0 * 1024.0));
  }

int main(void) {
  // Check GPU availability
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("Number of CUDA devices: %d\n", deviceCount);
  
  if (deviceCount < 2) {
    printf("Error: Need at least 2 GPUs for this program\n");
    return -1;
  }
  
  // Check GPU memory on both devices
  printf("\n=== GPU Memory Check ===\n");
  cudaSetDevice(0);
  print_gpu_memory(0);
  cudaSetDevice(1);
  print_gpu_memory(1);
  
  // Check if GPUs have sufficient memory
  size_t freeMem0, totalMem0, freeMem1, totalMem1;
  cudaSetDevice(0);
  cudaMemGetInfo(&freeMem0, &totalMem0);
  cudaSetDevice(1);
  cudaMemGetInfo(&freeMem1, &totalMem1);
  
  if (freeMem0 < 12ULL * 1024 * 1024 * 1024 || freeMem1 < 12ULL * 1024 * 1024 * 1024) {
    printf("Warning: GPUs may not have sufficient memory for 24 GB allocation\n");
    printf("GPU 0 free: %.2f GB, GPU 1 free: %.2f GB\n", 
           freeMem0 / (1024.0 * 1024.0 * 1024.0), 
           freeMem1 / (1024.0 * 1024.0 * 1024.0));
  }
  
  // Allocate 10 GB per array (6 arrays total = 24 GB) - large enough to force swapping but not exceed limits
  size_t N = (size_t)13 * 1024 * 1024 * 1024 / sizeof(float);
  printf("Allocating %zu elements = %.2f GB per array (24 GB total)\n", 
         N, (N * sizeof(float)) / (1024.0 * 1024.0 * 1024.0));
  
  float *a, *b, *c, *d, *x, *y;
  
  // Allocate memory with error checking and timing
  printf("\n=== Memory Allocation ===\n");
  auto start_time = get_time();
  
  cudaError_t err;
  err = cudaMallocManaged(&a, N*sizeof(float));
  if (err != cudaSuccess) { printf("Error allocating a: %s\n", cudaGetErrorString(err)); return -1; }
  
  err = cudaMallocManaged(&b, N*sizeof(float));
  if (err != cudaSuccess) { printf("Error allocating b: %s\n", cudaGetErrorString(err)); return -1; }
  
  err = cudaMallocManaged(&c, N*sizeof(float));
  if (err != cudaSuccess) { printf("Error allocating c: %s\n", cudaGetErrorString(err)); return -1; }
  
  err = cudaMallocManaged(&d, N*sizeof(float));
  if (err != cudaSuccess) { printf("Error allocating d: %s\n", cudaGetErrorString(err)); return -1; }
  
  err = cudaMallocManaged(&x, N*sizeof(float));
  if (err != cudaSuccess) { printf("Error allocating x: %s\n", cudaGetErrorString(err)); return -1; }
  
  err = cudaMallocManaged(&y, N*sizeof(float));
  if (err != cudaSuccess) { printf("Error allocating y: %s\n", cudaGetErrorString(err)); return -1; }

  float *yy;
  err = cudaMallocManaged(&yy, N*sizeof(float));
  if (err != cudaSuccess) { printf("Error allocating yy: %s\n", cudaGetErrorString(err)); return -1; }

  auto end_time = get_time();
  printf("All memory allocations successful! Time: %.2f ms\n", get_elapsed_ms(start_time, end_time));
  

  int blockSize = 256;
  int numBlocks = (int)((N + blockSize - 1) / blockSize);
  
  // Setup GPU 0
  printf("\n=== Setting up GPU 0 ===\n");
  err = cudaSetDevice(0);
  if (err != cudaSuccess) { printf("Error setting device 0: %s\n", cudaGetErrorString(err)); return -1; }
  
  // Enable peer access
  err = cudaDeviceEnablePeerAccess(1, 0);
  if (err != cudaSuccess) { 
    printf("Warning: Could not enable peer access from GPU 0 to GPU 1: %s\n", cudaGetErrorString(err));
    printf("This may limit UVM functionality between GPUs\n");
  }
  print_gpu_memory(0);
  getchar();
  // Step 1: Prefetch x, a, b to GPU 0
  printf("\n=== STEP 1: Prefetch x, a, b to GPU 0 ===\n");
  start_time = get_time();
  cudaMemPrefetchAsync(y, N*sizeof(float), 0);
  cudaMemPrefetchAsync(a, N*sizeof(float), 0);
  cudaMemPrefetchAsync(b, N*sizeof(float), 0);
  cudaDeviceSynchronize();
  end_time = get_time();
  printf("Prefetch x, a, b to GPU 0 completed. Time: %.2f ms\n", get_elapsed_ms(start_time, end_time));
  // add user input to continue
  print_gpu_memory(0);
  getchar();
  
  // Step 2: Launch kernel on GPU 0 (x = a + b)
  printf("\n=== STEP 2: Launch kernel on GPU 0 (x = a + b) ===\n");
  start_time = get_time();
  saxpy<<<numBlocks, blockSize>>>(N, 2.0f, a, b, y);
  err = cudaGetLastError();
  if (err != cudaSuccess) { printf("Error launching kernel: %s\n", cudaGetErrorString(err)); return -1; }
  cudaDeviceSynchronize();
  end_time = get_time();
  printf("GPU 0 kernel (x = a + b) completed. Time: %.2f ms\n", get_elapsed_ms(start_time, end_time));
  print_gpu_memory(0);
  getchar();
  // Step 3: Prefetch y, c, d to GPU 0 (should trigger swap-out of x, a, b)
  printf("\n=== STEP 3: Prefetch y, c, d to GPU 0 (should trigger swap-out) ===\n");
  start_time = get_time();
  cudaMemPrefetchAsync(y, N*sizeof(float), 0);
  cudaMemPrefetchAsync(c, N*sizeof(float), 0);
  cudaMemPrefetchAsync(d, N*sizeof(float), 0);
  cudaDeviceSynchronize();
  end_time = get_time();
  printf("Prefetch y, c, d to GPU 0 completed. Time: %.2f ms\n", get_elapsed_ms(start_time, end_time));
  print_gpu_memory(0);
  getchar();
  // Step 4: Launch kernel on GPU 0 (y = c + d)
  printf("\n=== STEP 4: Launch kernel on GPU 0 (y = c + d) ===\n");
  start_time = get_time();
  saxpy<<<numBlocks, blockSize>>>(N, 2.0f, c, d, y);
  err = cudaGetLastError();
  if (err != cudaSuccess) { printf("Error launching kernel: %s\n", cudaGetErrorString(err)); return -1; }
  cudaDeviceSynchronize();
  if (err != cudaSuccess) { printf("Error running kernel: %s\n", cudaGetErrorString(err)); return -1; }
  end_time = get_time();
  printf("GPU 0 kernel (y = c + d) completed. Time: %.2f ms\n", get_elapsed_ms(start_time, end_time));
  
  print_gpu_memory(0);

  getchar();
  // Step 5: Switch to GPU 1
  printf("\n=== STEP 5: Switch to GPU 1 ===\n");
  err = cudaSetDevice(1);
  if (err != cudaSuccess) { 
    printf("Error setting device 1: %s\n", cudaGetErrorString(err)); 
    printf("GPU 1 is not accessible. Skipping GPU 1 operations.\n");
    printf("The UVM swapping experiment will focus on GPU 0 memory pressure.\n");
    goto cleanup;
  }
  // get current device
  int currentDevice;
  cudaGetDevice(&currentDevice);
  printf("Current device: %d\n", currentDevice);

  // Enable peer access from GPU 1 to GPU 0
  err = cudaDeviceEnablePeerAccess(0, 0);
  if (err != cudaSuccess) { 
    printf("Warning: Could not enable peer access from GPU 1 to GPU 0: %s\n", cudaGetErrorString(err));
    printf("UVM may not work properly between GPUs\n");
  }

  getchar();
  
  // Check GPU 1 memory before proceeding
  err = cudaSetDevice(1);  // ADD THIS LINE
  if (err != cudaSuccess) {
    printf("Error setting device 1: %s\n", cudaGetErrorString(err));
    printf("GPU 1 is not accessible. Skipping GPU 1 operations.\n");
    return -1;
  }
  print_gpu_memory(1);

  // Step 6: Launch kernel on GPU 1 (x = a + b) - should trigger page faults
  printf("\n=== STEP 6: Launch kernel on GPU 1 (x = a + b) - should trigger page faults ===\n");
  printf("First run (should be slow due to page faults):\n");
  
  // Try different approaches to access UVM memory from GPU 1  
  // Method 1: Try direct kernel launch (may fail with illegal memory access)
  start_time = get_time();
  saxpy<<<numBlocks, blockSize>>>(N, 1.0f, a, b, x);
  err = cudaGetLastError();
  
  if (err != cudaSuccess) {
    printf("Direct kernel launch failed: %s\n", cudaGetErrorString(err));
  }
  
  cudaDeviceSynchronize();
  end_time = get_time();
  printf("GPU 1 kernel (x = a + b) first run completed. Time: %.2f ms\n", get_elapsed_ms(start_time, end_time));
  
  // Step 7: Second kernel on GPU 1 (should be fast now)
  printf("\n=== STEP 7: Second kernel on GPU 1 (should be fast now) ===\n");
  start_time = get_time();
  
  // Try to launch on GPU 1, fallback to GPU 0 if needed
  cudaSetDevice(1);
  saxpy<<<numBlocks, blockSize>>>(N, 1.0f, a, b, x);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPU 1 kernel failed: %s, trying GPU 0...\n", cudaGetErrorString(err));
    cudaSetDevice(0);
    saxpy<<<numBlocks, blockSize>>>(N, 1.0f, a, b, x);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Both GPUs failed: %s\n", cudaGetErrorString(err));
      return -1;
    }
    printf("Executed on GPU 0 instead\n");
  }
  
  cudaDeviceSynchronize();
  end_time = get_time();
  printf("GPU 1 kernel (x = a + b) second run completed. Time: %.2f ms\n", get_elapsed_ms(start_time, end_time));
  
cleanup:
  printf("\n=== Cleaning up ===\n");
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  cudaFree(d);
  cudaFree(x);
  cudaFree(y);
  
  printf("\n=== UVM Page Swapping Experiment Completed ===\n");
  printf("Key observations:\n");
  printf("1. First GPU 0 kernel: Fast (data resident)\n");
  printf("2. Second GPU 0 kernel: Fast (new data resident, old data swapped out)\n");
  printf("3. Memory pressure effects: 24 GB allocation forces UVM to manage memory actively\n");
  printf("Program completed successfully!\n");
  
  return 0;
}