use crate::runtime::GpuDialect;

pub(super) fn render(dialect: GpuDialect) -> String {
    let mut result = format!(
        "#include <{}>\n#include <stdint.h>\n#include <stddef.h>\n#include <stdio.h>\n#include <stdlib.h>\n",
        dialect.runtime_header()
    );
    result.push_str(&format!(
        r#"
__host__ __device__ inline void exp_assert(bool condition) {{
    if (!condition) {{
#if defined({})
        __builtin_trap();
#else
        abort();
#endif
    }}
}}
inline void exp_gpu_check({} error) {{
    if (error != {}) {{ fprintf(stderr, "GPU error: %s\n", {}(error)); abort(); }}
}}
"#,
        dialect.device_compile_guard(),
        dialect.error_type(),
        dialect.success_value(),
        dialect.error_string_fn()
    ));
    result.push_str(r#"
struct catena_mem_own_t { void* data; uint64_t len; };
struct catena_mem_ref_t { void* data; uint64_t len; };
struct exp_index { uint32_t x,y,z; };
struct exp_grid { dim3 blocks,threads; };
struct exp_block { exp_grid grid; exp_index index; };
struct exp_thread { exp_block block; exp_index index; };
template<class T> struct exp_global { T* data; uint32_t count; uint64_t bytes; };
template<class T> struct exp_shared_view { T* data; uint32_t count; };
template<size_t N> struct exp_layout {
    size_t offsets[N ? N : 1];
    uint32_t names[N ? N : 1], sizes[N ? N : 1], counts[N ? N : 1];
    size_t bytes;
};
__host__ __device__ inline exp_grid exp_make_grid(uint32_t bx,uint32_t by,uint32_t tx,uint32_t ty) {
    exp_assert(bx && by && tx && ty);
    return {dim3(bx,by,1),dim3(tx,ty,1)};
}
__host__ __device__ inline exp_index exp_grid_index(exp_thread t) {
    return {t.block.index.x*t.block.grid.threads.x+t.index.x,
            t.block.index.y*t.block.grid.threads.y+t.index.y,
            t.block.index.z*t.block.grid.threads.z+t.index.z};
}
__host__ __device__ inline uint32_t exp_row_major(uint32_t r,uint32_t c,uint32_t rows,uint32_t cols) {
    exp_assert(r < rows && c < cols && uint64_t(rows)*cols <= UINT32_MAX);
    return r*cols+c;
}
template<class T,class M> __host__ __device__ exp_global<T> exp_global_from_mem(M mem,uint32_t count) {
    exp_assert(uint64_t(count)*sizeof(T) <= mem.len);
    exp_assert(!count || mem.data);
    exp_assert(reinterpret_cast<uintptr_t>(mem.data)%alignof(T)==0);
    return {static_cast<T*>(mem.data),count,mem.len};
}
template<size_t N> __host__ __device__ exp_layout<N+1> exp_add_slot(exp_layout<N> rest,uint32_t name,uint32_t size,uint32_t count) {
    exp_layout<N+1> result{};
    for (size_t i=0;i<N;++i) {
        exp_assert(rest.names[i]!=name);
        result.offsets[i]=rest.offsets[i]; result.names[i]=rest.names[i];
        result.sizes[i]=rest.sizes[i]; result.counts[i]=rest.counts[i];
    }
    exp_assert(rest.bytes <= SIZE_MAX-15);
    size_t offset=(rest.bytes+15)&~size_t(15);
    exp_assert(size!=0 && count <= (SIZE_MAX-offset)/size);
    result.offsets[N]=offset; result.names[N]=name; result.sizes[N]=size; result.counts[N]=count;
    result.bytes=offset+size_t(count)*size;
    return result;
}
template<class T,size_t N> __device__ exp_shared_view<T> exp_slot(unsigned char* base,exp_layout<N> layout,uint32_t name,uint32_t count) {
    for (size_t i=0;i<N;++i) if(layout.names[i]==name) {
        exp_assert(layout.sizes[i]==sizeof(T) && layout.counts[i]==count);
        return {reinterpret_cast<T*>(base+layout.offsets[i]),count};
    }
    exp_assert(false); return {nullptr,0};
}
"#);
    result
}
