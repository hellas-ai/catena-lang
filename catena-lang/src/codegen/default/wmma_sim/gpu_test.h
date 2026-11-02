// HIP/CUDA runtime aliases used only by the stress executable.
#define WMMA_CUDA 0
#if defined(__CUDACC__) && !defined(__HIPCC__)
#undef WMMA_CUDA
#define WMMA_CUDA 1
#include <cuda_runtime.h>
// Keep the existing HIP call sites; direct nvcc builds need no HIP installation.
#define hipError_t cudaError_t
#define hipSuccess cudaSuccess
#define hipGetErrorString cudaGetErrorString
#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipGetDeviceCount cudaGetDeviceCount
#define hipDeviceProp_t cudaDeviceProp
#define hipGetDeviceProperties cudaGetDeviceProperties
#define hipSetDevice cudaSetDevice
#define hipMemcpy cudaMemcpy
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipGetLastError cudaGetLastError
#else
#include <hip/hip_runtime.h>
#endif

#if WMMA_CUDA
#define WMMA_AMD_GPU 0
#else
#define WMMA_AMD_GPU 1
#endif
