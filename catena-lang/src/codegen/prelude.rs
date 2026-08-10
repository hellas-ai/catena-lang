use crate::codegen::GpuDialect;

pub fn render_gpu_prelude(dialect: GpuDialect) -> String {
    let buffer_load = render_buffer_load(dialect);
    let bf16_support = render_bf16_support(dialect);
    format!(
        r#"#include <{runtime_header}>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

{bf16_support}

typedef uint8_t catena_unit_t;
typedef uint8_t catena_gpu_state_t;

typedef struct {{
    uint32_t x;
    uint32_t y;
    uint32_t z;
}} catena_dim3_t;

typedef struct {{
    uint64_t thread_id;
}} catena_gpu_env_t;

typedef struct {{
    catena_dim3_t grid_dim;
    catena_dim3_t block_dim;
}} catena_launch_params_t;

typedef struct {{
    void *data;
    uint64_t len;
}} catena_mem_own_t;

typedef struct {{
    void *data;
    uint64_t len;
}} catena_mem_ref_t;

typedef struct {{
    void *data;
    uint64_t len;
}} catena_gpu_buf_t;

__host__ __device__ static inline void catena_assert(uint8_t condition) {{
    if (!condition) {{
#ifndef {device_compile_guard}
        fprintf(stderr, "catena assertion failed\n");
        fflush(stderr);
#endif
        __builtin_trap();
    }}
}}

#ifndef {device_compile_guard}
__host__ static inline void catena_host_gpu_check({error_type} err) {{
    if (err != {success_value}) {{
        fprintf(stderr, "catena GPU error: %s\n", {error_string_fn}(err));
        fflush(stderr);
        __builtin_trap();
    }}
}}

__host__ static inline void catena_host_buffer_free(void *data) {{
    if (data != nullptr) {{
        catena_host_gpu_check({device_free_fn}(data));
    }}
}}

#endif

__host__ __device__ static inline uint64_t catena_launch_len(catena_launch_params_t params) {{
    return (uint64_t)params.grid_dim.x * params.grid_dim.y * params.grid_dim.z
        * params.block_dim.x * params.block_dim.y * params.block_dim.z;
}}

__host__ __device__ static inline float catena_u32_bitcast_f32(uint32_t bits) {{
    union {{
        uint32_t u;
        float f;
    }} value;
    value.u = bits;
    return value.f;
}}

__host__ __device__ static inline uint32_t catena_f32_bitcast_u32(float value) {{
    union {{
        uint32_t u;
        float f;
    }} bits;
    bits.f = value;
    return bits.u;
}}

{buffer_load}

"#,
        runtime_header = dialect.runtime_header(),
        device_compile_guard = dialect.device_compile_guard(),
        error_type = dialect.error_type(),
        success_value = dialect.success_value(),
        error_string_fn = dialect.error_string_fn(),
        device_free_fn = dialect.device_free_fn(),
        buffer_load = buffer_load,
        bf16_support = bf16_support,
    )
}

fn render_bf16_support(dialect: GpuDialect) -> &'static str {
    match dialect {
        GpuDialect::Hip => {
            r#"#include <hip/hip_bfloat16.h>
typedef hip_bfloat16 catena_bf16_t;
static_assert(sizeof(catena_bf16_t) == 2, "BF16 must occupy two bytes");
static_assert(alignof(catena_bf16_t) == 2, "BF16 must have two-byte alignment");

__host__ __device__ static inline catena_bf16_t catena_bf16_from_f32(float value) {
    return catena_bf16_t(value);
}

__host__ __device__ static inline float catena_bf16_to_f32(catena_bf16_t value) {
    return (float)value;
}

__host__ __device__ static inline catena_bf16_t catena_u16_bitcast_bf16(uint16_t bits) {
    catena_bf16_t value;
    value.data = bits;
    return value;
}

__host__ __device__ static inline uint16_t catena_bf16_bitcast_u16(catena_bf16_t value) {
    return value.data;
}"#
        }
        GpuDialect::Cuda => {
            r#"#include <cuda_bf16.h>
typedef __nv_bfloat16 catena_bf16_t;
static_assert(sizeof(catena_bf16_t) == 2, "BF16 must occupy two bytes");
static_assert(alignof(catena_bf16_t) == 2, "BF16 must have two-byte alignment");

__host__ __device__ static inline catena_bf16_t catena_bf16_from_f32(float value) {
    return __float2bfloat16_rn(value);
}

__host__ __device__ static inline float catena_bf16_to_f32(catena_bf16_t value) {
    return __bfloat162float(value);
}

__host__ __device__ static inline catena_bf16_t catena_u16_bitcast_bf16(uint16_t bits) {
    return __ushort_as_bfloat16(bits);
}

__host__ __device__ static inline uint16_t catena_bf16_bitcast_u16(catena_bf16_t value) {
    return __bfloat16_as_ushort(value);
}"#
        }
    }
}

fn render_buffer_load(dialect: GpuDialect) -> String {
    format!(
        r#"template <typename T>
__host__ __device__ static inline T catena_device_buffer_load(const T *buffer, uint64_t index) {{
#ifdef {device_compile_guard}
    return buffer[index];
#else
    // This is a synchronous scalar copy. Host loops such as reducec can issue
    // one device-to-host transfer per element and should eventually be kernelized.
    T value;
    catena_host_gpu_check({memcpy_fn}(&value, buffer + index, sizeof(T), {memcpy_device_to_host}));
    return value;
#endif
}}"#,
        device_compile_guard = dialect.device_compile_guard(),
        memcpy_fn = dialect.memcpy_fn(),
        memcpy_device_to_host = dialect.memcpy_device_to_host(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_gpu_check_is_host_only() {
        let prelude = render_gpu_prelude(GpuDialect::Hip);

        assert!(
            prelude.contains(
                "#ifndef __HIP_DEVICE_COMPILE__\n__host__ static inline void catena_host_gpu_check(hipError_t err)"
            )
        );
        assert!(!prelude.contains("catena_gpu_check"));
        assert!(!prelude.contains("__device__ static inline void catena_host_gpu_check"));
    }

    #[test]
    fn buffer_free_uses_the_selected_host_runtime() {
        let hip = render_gpu_prelude(GpuDialect::Hip);
        assert!(hip.contains("catena_host_gpu_check(hipFree(data));"));

        let cuda = render_gpu_prelude(GpuDialect::Cuda);
        assert!(cuda.contains("catena_host_gpu_check(cudaFree(data));"));
    }

    #[test]
    fn hip_buffer_load_test() {
        let prelude = render_gpu_prelude(GpuDialect::Hip);

        assert!(prelude.contains("#ifdef __HIP_DEVICE_COMPILE__\n    return buffer[index];"));
        assert!(prelude.contains(
            "catena_host_gpu_check(hipMemcpy(&value, buffer + index, sizeof(T), hipMemcpyDeviceToHost));"
        ));
    }

    #[test]
    fn cuda_buffer_load_test() {
        let prelude = render_gpu_prelude(GpuDialect::Cuda);

        assert!(prelude.contains("#ifdef __CUDA_ARCH__\n    return buffer[index];"));
        assert!(prelude.contains(
            "catena_host_gpu_check(cudaMemcpy(&value, buffer + index, sizeof(T), cudaMemcpyDeviceToHost));"
        ));
    }

    #[test]
    fn bf16_support_uses_each_dialects_native_type() {
        let hip = render_gpu_prelude(GpuDialect::Hip);
        assert!(hip.contains("#include <hip/hip_bfloat16.h>"));
        assert!(hip.contains("typedef hip_bfloat16 catena_bf16_t;"));
        assert!(hip.contains("return catena_bf16_t(value);"));
        assert!(hip.contains("value.data = bits;"));

        let cuda = render_gpu_prelude(GpuDialect::Cuda);
        assert!(cuda.contains("#include <cuda_bf16.h>"));
        assert!(cuda.contains("typedef __nv_bfloat16 catena_bf16_t;"));
        assert!(cuda.contains("return __float2bfloat16_rn(value);"));
        assert!(cuda.contains("return __ushort_as_bfloat16(bits);"));
    }
}
