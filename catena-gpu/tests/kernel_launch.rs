use catena_gpu::{
    codegen::{GpuDialect, gpu::render_modules},
    compile::compile,
};
use metacat::theory::RawTheorySet;

const LINEAR_SCHEDULING: &str = include_str!("cases/linear_scheduling.hex");

#[test]
fn perfect_tiling_generates_linear_kernel_launch() -> anyhow::Result<()> {
    let parameters = LaunchParameters {
        buffer_size: 8,
        grid_x: 2,
        block_x: 4,
    };
    assert_eq!(parameters.launched_threads(), parameters.buffer_size);
    check_launch_case(parameters, GpuDialect::Hip)
}

#[test]
fn predicated_tiling_generates_linear_kernel_launch() -> anyhow::Result<()> {
    let parameters = LaunchParameters {
        buffer_size: 7,
        grid_x: 2,
        block_x: 4,
    };
    assert!(parameters.launched_threads() > parameters.buffer_size);
    check_launch_case(parameters, GpuDialect::Cuda)
}

struct LaunchParameters {
    buffer_size: u64,
    grid_x: u64,
    block_x: u64,
}

impl LaunchParameters {
    fn launched_threads(&self) -> u64 {
        self.grid_x
            .checked_mul(self.block_x)
            .expect("test launch dimensions should not overflow")
    }
}

fn check_launch_case(parameters: LaunchParameters, dialect: GpuDialect) -> anyhow::Result<()> {
    assert!(parameters.buffer_size <= parameters.launched_threads());
    let generated = compile_sources([LINEAR_SCHEDULING], dialect)?;
    assert!(generated.contains("__global__"));
    assert!(generated.contains("global_linear_id"));
    assert!(generated.contains("catena_gpu_synchronize"));
    assert!(generated.contains(".global_linear_id =="));
    assert!(generated.contains("uint64_t *buffer"));
    Ok(())
}

fn compile_sources(
    sources: impl IntoIterator<Item = &'static str>,
    dialect: GpuDialect,
) -> anyhow::Result<String> {
    let raw = RawTheorySet::from_texts(catena_gpu::stdlib::sources().chain(sources))?;
    let report = compile(raw)?;
    Ok(render_modules(
        report.gpu_modules.as_ref().expect("codegen modules"),
        dialect,
    )?)
}
