use crate::codegen::GpuDialect;

pub fn render_gpu_prelude(dialect: GpuDialect) -> String {
    format!(
        r#"#include <{runtime_header}>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

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
}} catena_mem_t;

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

#endif

__host__ __device__ static inline uint64_t catena_launch_len(catena_launch_params_t params) {{
    return (uint64_t)params.grid_dim.x * params.grid_dim.y * params.grid_dim.z
        * params.block_dim.x * params.block_dim.y * params.block_dim.z;
}}

__host__ __device__ static inline void bool_not(uint8_t arg0, uint8_t *out1) {{
    *out1 = !arg0;
}}

__host__ __device__ static inline void bool_or(uint8_t arg0, uint8_t arg1, uint8_t *out2) {{
    *out2 = arg0 || arg1;
}}

__host__ __device__ static inline void bool_and(uint8_t arg0, uint8_t arg1, uint8_t *out2) {{
    *out2 = arg0 && arg1;
}}

__host__ __device__ static inline void bool_id(uint8_t arg0, uint8_t *out1) {{
    *out1 = arg0;
}}

__host__ __device__ static inline void bool_copy(uint8_t arg0, uint8_t *out1, uint8_t *out2) {{
    *out1 = arg0;
    *out2 = arg0;
}}

__host__ __device__ static inline void bool_li(uint8_t arg0, uint8_t *out1) {{
    *out1 = arg0;
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

"#,
        runtime_header = dialect.runtime_header(),
        device_compile_guard = dialect.device_compile_guard(),
        error_type = dialect.error_type(),
        success_value = dialect.success_value(),
        error_string_fn = dialect.error_string_fn(),
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
}
