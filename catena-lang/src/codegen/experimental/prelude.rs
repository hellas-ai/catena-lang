use crate::runtime::GpuDialect;

pub(super) fn render(dialect: GpuDialect) -> String {
    let mut result = format!(
        "#include <{}>\n#include <stdint.h>\n#include <stddef.h>\n#include <stdio.h>\n#include <stdlib.h>\n",
        dialect.runtime_header()
    );
    result.push_str(&format!(
        r#"
__host__ __device__ inline void catena_assert(bool condition) {{
    if (!condition) {{
#if defined({})
        __builtin_trap();
#else
        abort();
#endif
    }}
}}
inline void catena_gpu_check({} error) {{
    if (error != {}) {{ fprintf(stderr, "GPU error: %s\n", {}(error)); abort(); }}
}}
"#,
        dialect.device_compile_guard(),
        dialect.error_type(),
        dialect.success_value(),
        dialect.error_string_fn()
    ));
    result.push_str(r#"
__host__ __device__ inline float catena_u32_bitcast_f32(uint32_t bits) {
    union { uint32_t u; float f; } value;
    value.u = bits;
    return value.f;
}
struct catena_mem_own_t { void* data; uint64_t len; };
struct catena_mem_ref_t { void* data; uint64_t len; };
struct catena_index { uint32_t x,y,z; };
struct catena_grid { dim3 blocks,threads; };
struct catena_block { catena_grid grid; catena_index index; };
struct catena_thread { catena_block block; catena_index index; };
template<class T> struct catena_global { T* data; uint32_t count; uint64_t bytes; };
template<class T> struct catena_shared_view { T* data; uint32_t count; };
template<size_t N> struct catena_layout {
    size_t offsets[N ? N : 1];
    uint32_t names[N ? N : 1], sizes[N ? N : 1], counts[N ? N : 1];
    size_t bytes;
};
__host__ __device__ inline catena_grid catena_make_grid(uint32_t bx,uint32_t by,uint32_t tx,uint32_t ty) {
    catena_assert(bx && by && tx && ty);
    return {dim3(bx,by,1),dim3(tx,ty,1)};
}
__host__ __device__ inline catena_index catena_grid_index(catena_thread t) {
    return {t.block.index.x*t.block.grid.threads.x+t.index.x,
            t.block.index.y*t.block.grid.threads.y+t.index.y,
            t.block.index.z*t.block.grid.threads.z+t.index.z};
}
__host__ __device__ inline uint32_t catena_row_major(uint32_t r,uint32_t c,uint32_t rows,uint32_t cols) {
    catena_assert(r < rows && c < cols && uint64_t(rows)*cols <= UINT32_MAX);
    return r*cols+c;
}
template<class T,class M> __host__ __device__ catena_global<T> catena_global_from_mem(M mem,uint32_t count) {
    catena_assert(uint64_t(count)*sizeof(T) <= mem.len);
    catena_assert(!count || mem.data);
    catena_assert(reinterpret_cast<uintptr_t>(mem.data)%alignof(T)==0);
    return {static_cast<T*>(mem.data),count,mem.len};
}
template<size_t N> __host__ __device__ catena_layout<N+1> catena_add_slot(catena_layout<N> rest,uint32_t name,uint32_t size,uint32_t count) {
    catena_layout<N+1> result{};
    for (size_t i=0;i<N;++i) {
        catena_assert(rest.names[i]!=name);
        result.offsets[i]=rest.offsets[i]; result.names[i]=rest.names[i];
        result.sizes[i]=rest.sizes[i]; result.counts[i]=rest.counts[i];
    }
    catena_assert(rest.bytes <= SIZE_MAX-15);
    size_t offset=(rest.bytes+15)&~size_t(15);
    catena_assert(size!=0 && count <= (SIZE_MAX-offset)/size);
    result.offsets[N]=offset; result.names[N]=name; result.sizes[N]=size; result.counts[N]=count;
    result.bytes=offset+size_t(count)*size;
    return result;
}
template<class T,size_t N> __device__ catena_shared_view<T> catena_slot(unsigned char* base,catena_layout<N> layout,uint32_t name,uint32_t count) {
    for (size_t i=0;i<N;++i) if(layout.names[i]==name) {
        catena_assert(layout.sizes[i]==sizeof(T) && layout.counts[i]==count);
        return {reinterpret_cast<T*>(base+layout.offsets[i]),count};
    }
    catena_assert(false); return {nullptr,0};
}
"#);
    result
}
