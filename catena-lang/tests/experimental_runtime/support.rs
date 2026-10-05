use std::path::Path;

use anyhow::Context;
use catena_lang::{
    codegen::CodegenKind,
    compile::compile,
    report::CompileReport,
    runtime::{Artifact, GpuDialect, Runtime},
    stdlib::BundleRegistry,
};
use metacat::theory::RawTheorySet;

pub fn runtime_with_fixture(fixture: &str) -> anyhow::Result<(Runtime, Artifact)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dialect = configured_gpu_dialect()?;
    let mut registry = BundleRegistry::new(&[])?;
    let name = registry.add_directory(root.join("stdlib/experimental"))?;
    let files = registry.resolve(&[&name])?;
    let path = root
        .join("tests/experimental_runtime/fixtures")
        .join(fixture);
    let source = std::fs::read_to_string(&path)
        .with_context(|| format!("reading runtime fixture {}", path.display()))?;
    let raw = RawTheorySet::from_texts(
        files
            .iter()
            .map(|file| file.source.as_ref())
            .chain([source.as_str()]),
    )?;
    let mut report = CompileReport::new(raw);
    let module = compile(&mut report, CodegenKind::Experimental, dialect)
        .with_context(|| format!("compiling experimental runtime fixture {fixture}"))?;
    let mut runtime = Runtime::new(dialect)?;
    let artifact = runtime.load(module)?;
    Ok((runtime, artifact))
}

fn configured_gpu_dialect() -> anyhow::Result<GpuDialect> {
    match std::env::var("CATENA_GPU_DIALECT").as_deref() {
        Ok("hip") | Err(std::env::VarError::NotPresent) => Ok(GpuDialect::Hip),
        Ok("cuda") => Ok(GpuDialect::Cuda),
        Ok(value) => anyhow::bail!(
            "invalid GPU dialect `{value}` in CATENA_GPU_DIALECT; expected `hip` or `cuda`"
        ),
        Err(std::env::VarError::NotUnicode(value)) => {
            anyhow::bail!("invalid GPU dialect in CATENA_GPU_DIALECT: non-Unicode value {value:?}")
        }
    }
}
