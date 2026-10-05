//! HIP/CUDA spellings shared by code generators.

use super::GpuDialect;

impl GpuDialect {
    pub fn runtime_header(self) -> &'static str {
        match self {
            Self::Hip => "hip/hip_runtime.h",
            Self::Cuda => "cuda_runtime.h",
        }
    }

    pub fn error_type(self) -> &'static str {
        match self {
            Self::Hip => "hipError_t",
            Self::Cuda => "cudaError_t",
        }
    }

    pub fn success_value(self) -> &'static str {
        match self {
            Self::Hip => "hipSuccess",
            Self::Cuda => "cudaSuccess",
        }
    }

    pub fn error_string_fn(self) -> &'static str {
        match self {
            Self::Hip => "hipGetErrorString",
            Self::Cuda => "cudaGetErrorString",
        }
    }

    pub(crate) fn device_alloc_fn(self) -> &'static str {
        match self {
            Self::Hip => "hipMalloc",
            Self::Cuda => "cudaMalloc",
        }
    }

    pub(crate) fn device_free_fn(self) -> &'static str {
        match self {
            Self::Hip => "hipFree",
            Self::Cuda => "cudaFree",
        }
    }

    pub fn synchronize_fn(self) -> &'static str {
        match self {
            Self::Hip => "hipDeviceSynchronize",
            Self::Cuda => "cudaDeviceSynchronize",
        }
    }

    pub(crate) fn memcpy_fn(self) -> &'static str {
        match self {
            Self::Hip => "hipMemcpy",
            Self::Cuda => "cudaMemcpy",
        }
    }

    pub(crate) fn memcpy_device_to_host(self) -> &'static str {
        match self {
            Self::Hip => "hipMemcpyDeviceToHost",
            Self::Cuda => "cudaMemcpyDeviceToHost",
        }
    }

    pub fn device_compile_guard(self) -> &'static str {
        match self {
            Self::Hip => "__HIP_DEVICE_COMPILE__",
            Self::Cuda => "__CUDA_ARCH__",
        }
    }
}
