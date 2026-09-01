use catena_gpu::{
    codegen::GpuDialect,
    runtime::{Artifact, Runtime, Value, load_sources},
    stdlib,
};

mod cases;

fn runtime_with(source: &'static str) -> anyhow::Result<(Runtime, Artifact)> {
    let mut runtime = Runtime::new(configured_gpu_dialect()?)?;
    let artifact = load_sources(&mut runtime, stdlib::sources().chain([source]))?;
    Ok((runtime, artifact))
}

fn configured_gpu_dialect() -> anyhow::Result<GpuDialect> {
    match std::env::var("CATENA_GPU_DIALECT") {
        Ok(value) if value == "cuda" => Ok(GpuDialect::Cuda),
        Ok(value) if value == "hip" => Ok(GpuDialect::Hip),
        Err(std::env::VarError::NotPresent) => Ok(GpuDialect::Hip),
        Ok(value) => anyhow::bail!("invalid CATENA_GPU_DIALECT `{value}`"),
        Err(std::env::VarError::NotUnicode(value)) => {
            anyhow::bail!("non-Unicode CATENA_GPU_DIALECT value: {value:?}")
        }
    }
}
