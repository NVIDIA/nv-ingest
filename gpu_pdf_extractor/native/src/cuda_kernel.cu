// Phase-0 CUDA proof: a trivial kernel exercised through the Python module.
// Replaced in P2 by the real batched rasterizer.
#include <cuda_runtime.h>
#include <string>

__global__ void add_kernel(const int* a, const int* b, int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

// Runs a[i]+b[i] on the GPU for a single element; returns the device result.
// Proves nvcc-compiled device code launches and round-trips from inside the .so.
int cuda_add(int a, int b) {
  int *da, *db, *dout, hout = -1;
  cudaMalloc(&da, sizeof(int)); cudaMalloc(&db, sizeof(int)); cudaMalloc(&dout, sizeof(int));
  cudaMemcpy(da, &a, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(db, &b, sizeof(int), cudaMemcpyHostToDevice);
  add_kernel<<<1, 1>>>(da, db, dout, 1);
  cudaDeviceSynchronize();
  cudaMemcpy(&hout, dout, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(da); cudaFree(db); cudaFree(dout);
  return hout;
}

std::string cuda_device_info() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) return "cuda: no devices / error";
  cudaDeviceProp p{};
  cudaGetDeviceProperties(&p, 0);
  return "cuda devices=" + std::to_string(n) + " dev0=" + p.name +
         " sm_" + std::to_string(p.major) + std::to_string(p.minor);
}
