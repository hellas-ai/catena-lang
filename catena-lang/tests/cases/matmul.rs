use catena_lang::{
    codegen::{GpuDialect, gpu::render_modules},
    compile::compile,
    stdlib,
};
use metacat::theory::RawTheorySet;

const INNER_SOURCE: &str = include_str!("matmul/inner.hex");

#[test]
#[ignore = "blocked by cross-file matmul-f32-inner wrapper compilation"]
fn f32_inner_generates_gpu_code() -> anyhow::Result<()> {
    let raw = RawTheorySet::from_texts(stdlib::sources().chain([INNER_SOURCE]))?;
    let report = compile(raw)?;
    let modules = report
        .gpu_modules
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("compile report did not contain GPU modules"))?;
    let rendered = render_modules(modules, GpuDialect::Hip)?;

    assert!(rendered.contains("program_matmul_test_f32_inner_ones"));
    assert!(rendered.contains("catena_reduce"));
    Ok(())
}
