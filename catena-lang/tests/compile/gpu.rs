use catena_lang::codegen::{GpuDialect, gpu::render_modules};

const TILED_MATMUL_SOURCE: &str = include_str!("../cases/gpu/tiled_matmul.hex");
const NAIVE_MATMUL_SOURCE: &str = include_str!("../cases/matrix/matmul.hex");

#[test]
fn tiled_matmul_converts_and_renders_cooperative_loads() -> anyhow::Result<()> {
    let report = super::compile_with_sources([TILED_MATMUL_SOURCE])?;
    let modules = report
        .gpu_modules
        .as_ref()
        .expect("successful compilation should produce GPU modules");

    assert!(modules.values().any(|module| {
        module
            .entry
            .assignments
            .iter()
            .any(|assignment| assignment.op.as_str() == "gpu.shared.row-major.cooperative-loadc")
    }));

    let source = render_modules(modules, GpuDialect::Hip)?;
    assert!(source.contains("(uint64_t)threadIdx.y"));
    assert!(source.contains("(uint64_t)threadIdx.x"));
    assert!(source.contains("__syncthreads();"));
    Ok(())
}

#[test]
fn naive_matmul_has_distinct_ordinary_and_gpu_materialization() -> anyhow::Result<()> {
    let report = super::compile_with_sources([NAIVE_MATMUL_SOURCE])?;
    let modules = report
        .gpu_modules
        .as_ref()
        .expect("successful compilation should produce GPU modules");

    assert!(modules.values().any(|module| {
        module
            .entry
            .assignments
            .iter()
            .any(|assignment| assignment.op.as_str() == "materializec")
    }));
    assert!(modules.values().any(|module| {
        module
            .entry
            .assignments
            .iter()
            .any(|assignment| assignment.op.as_str() == "gpu.materialize")
    }));

    let source = render_modules(modules, GpuDialect::Cuda)?;
    assert!(source.contains("for (uint64_t"));
    assert!(source.contains("__global__ void materialize_program_gpu_naive_matmul_via_mem"));
    assert!(!source.contains("__global__ void materialize_program_matmul_2x2_via_mem"));
    Ok(())
}
