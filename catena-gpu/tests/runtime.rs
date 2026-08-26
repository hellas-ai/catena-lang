use catena_gpu::{
    codegen::GpuDialect,
    runtime::{Runtime, Value},
    stdlib,
};

mod cases;

#[test]
fn scalar_program_executes() -> anyhow::Result<()> {
    let dialect = match std::env::var("CATENA_GPU_DIALECT") {
        Ok(value) if value == "cuda" => GpuDialect::Cuda,
        Ok(value) if value == "hip" => GpuDialect::Hip,
        Err(std::env::VarError::NotPresent) => GpuDialect::Hip,
        Ok(value) => anyhow::bail!("invalid CATENA_GPU_DIALECT `{value}`"),
        Err(std::env::VarError::NotUnicode(value)) => {
            anyhow::bail!("non-Unicode CATENA_GPU_DIALECT value: {value:?}")
        }
    };
    let mut runtime = Runtime::new(dialect)?;
    let artifact = runtime.load_sources(stdlib::sources().chain([cases::MINIMAL]))?;
    let [value] = runtime.exec(&artifact, "add", [20_u64.into(), 22_u64.into()])?;
    assert!(matches!(value, Value::U64(42)));
    Ok(())
}
