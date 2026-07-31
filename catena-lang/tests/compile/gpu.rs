use catena_lang::codegen::{
    GpuDialect,
    gpu::{render_module, render_modules},
};

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
fn ordinary_naive_matmul_uses_sequential_materialize_without_a_kernel() -> anyhow::Result<()> {
    let report = super::compile_with_sources([NAIVE_MATMUL_SOURCE])?;
    let modules = report
        .gpu_modules
        .as_ref()
        .expect("successful compilation should produce GPU modules");
    let ordinary = modules
        .values()
        .find(|module| {
            module
                .source_name
                .as_ref()
                .is_some_and(|name| name.as_str() == "matmul-2x2-via-mem")
        })
        .expect("ordinary naive matmul should have a generated module");

    assert!(
        ordinary
            .entry
            .assignments
            .iter()
            .any(|assignment| assignment.op.as_str() == "materializec")
    );
    assert!(
        ordinary
            .entry
            .assignments
            .iter()
            .all(|assignment| assignment.op.as_str() != "gpu.materialize")
    );

    let source = render_module(ordinary, GpuDialect::Cuda)?;
    assert!(source.contains("for (uint64_t"));
    assert!(!source.contains("__global__"));
    assert!(!source.contains("<<<"));
    assert!(!source.contains("DeviceSynchronize"));
    Ok(())
}

#[test]
fn gpu_naive_matmul_uses_gpu_materialize_and_a_kernel() -> anyhow::Result<()> {
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
            .any(|assignment| assignment.op.as_str() == "gpu.materialize")
    }));

    let source = render_modules(modules, GpuDialect::Cuda)?;
    assert!(source.contains("__global__ void materialize_program_gpu_naive_matmul_via_mem"));
    assert!(source.contains("materialize_program_gpu_naive_matmul_via_mem"));
    assert!(source.contains("<<<"));
    assert!(source.contains("cudaDeviceSynchronize"));
    Ok(())
}
