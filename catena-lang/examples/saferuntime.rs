use std::{env, path::PathBuf};

use catena_lang::{
    codegen::GpuDialect,
    runtime::{SafeExecError, SafeRuntime, Value, run_safe_runtime_child_if_requested},
    stdlib,
};

const GPU_DIALECT_ENV: &str = "CATENA_GPU_DIALECT";

fn main() -> anyhow::Result<()> {
    // If child mode is enabled after being spawned by the parent, run the child loop.
    // Otherwise, continue with the normal runtime execution as a parent process.
    if run_safe_runtime_child_if_requested()? {
        return Ok(());
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let runtime = SafeRuntime::new(
        stdlib::paths_from(&root).chain([root.join("examples/example.hex")]),
        configured_gpu_dialect()?,
    )?;

    let [result] = runtime.exec("two-times-two", [])?;
    let Value::U64(result) = result else {
        anyhow::bail!("two-times-two returned non-u64 value: {result:?}");
    };
    println!("two-times-two: {result}");
    anyhow::ensure!(result == 4, "two-times-two returned {result}, expected 4");

    let [] = runtime.exec("require-true", [true.into()])?;

    match runtime.exec::<1, 0>("require-true", [false.into()]) {
        Err(SafeExecError::ChildTerminated { status, stderr }) => {
            anyhow::ensure!(!status.success(), "asserting child exited successfully");
            anyhow::ensure!(
                stderr.contains("catena assertion failed"),
                "asserting child did not report the expected failure: {stderr:?}"
            );
        }
        Err(error) => anyhow::bail!("require-true(false) returned the wrong error: {error}"),
        Ok([]) => anyhow::bail!("require-true(false) unexpectedly returned successfully"),
    }
    println!("require-true(false): child assertion isolated");

    Ok(())
}

fn configured_gpu_dialect() -> anyhow::Result<GpuDialect> {
    match env::var(GPU_DIALECT_ENV).as_deref() {
        Ok("hip") | Err(env::VarError::NotPresent) => Ok(GpuDialect::Hip),
        Ok("cuda") => Ok(GpuDialect::Cuda),
        Ok(value) => anyhow::bail!(
            "invalid GPU dialect `{value}` in {GPU_DIALECT_ENV}; expected `hip` or `cuda`"
        ),
        Err(env::VarError::NotUnicode(value)) => anyhow::bail!(
            "invalid GPU dialect in {GPU_DIALECT_ENV}: non-Unicode value {:?}",
            value
        ),
    }
}
