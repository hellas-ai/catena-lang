use catena_lang::codegen::{GpuDialect, gpu::render_modules};

use super::compile_with_sources;

const TILED_MATMUL: &str = include_str!("../cases/matrix/tiled_matmul.hex");

/// Exercise the complete compiler and both source renderers without requiring
/// a GPU compiler or device. Runtime execution is covered separately when the
/// `runtime-tests` feature and a GPU toolchain are available.
#[test]
fn tiled_matmul_reaches_hip_and_cuda_source() {
    let report =
        compile_with_sources([TILED_MATMUL]).expect("tiled matmul should compile into GPU modules");
    let modules = report
        .gpu_modules
        .as_ref()
        .expect("codegen should produce GPU modules");

    for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
        let source = render_modules(modules, dialect)
            .unwrap_or_else(|error| panic!("{dialect:?} rendering failed: {error}"));
        assert!(source.contains("__global__ void parallel_materialize_"));
        assert!(source.contains("catena_gpu_grid_host_t"));
        assert!(source.contains("catena_gpu_block_worker_t"));
        assert!(source.contains(".index %"));
        assert!(source.contains(".launch.block_dim.x +"));
        assert!(source.contains("catena_block_barrier()"));
    }
}
