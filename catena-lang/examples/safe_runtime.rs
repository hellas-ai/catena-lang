use std::env;

use catena_lang::{
    codegen::GpuDialect,
    runtime::Value,
    safe_runtime::{SafeExecError, SafeRuntime, run_safe_runtime_child_if_requested},
    stdlib,
};

const GPU_DIALECT_ENV: &str = "CATENA_GPU_DIALECT";

const MEMORY_SOURCE: &str = r#"
(def program safe-mem-own-identity : (cap.own mem) -> (cap.own mem) = [memory])
"#;

fn main() -> anyhow::Result<()> {
    // If child mode is enabled after being spawned by the parent, run the child loop.
    // Otherwise, continue with the normal runtime execution as a parent process.
    if run_safe_runtime_child_if_requested()? {
        return Ok(());
    }

    let runtime = SafeRuntime::from_sources(
        stdlib::sources().chain([include_str!("example.hex"), MEMORY_SOURCE]),
        configured_gpu_dialect()?,
    )?;

    let [result] = runtime.exec("two-times-two", [])?;
    let Value::U64(result) = result else {
        anyhow::bail!("two-times-two returned non-u64 value: {result:?}");
    };
    println!("two-times-two: {result}");
    anyhow::ensure!(result == 4, "two-times-two returned {result}, expected 4");

    let [] = runtime.exec("require-true", [true.into()])?;

    let input = runtime.mem_u64(&[3, 5, 8, 13])?;
    let [output] = runtime.exec("safe-mem-own-identity", [input.into()])?;
    let Value::MemOwn(output) = output else {
        anyhow::bail!("safe-mem-own-identity returned non-owned memory: {output:?}");
    };
    anyhow::ensure!(output.try_to_u64_vec()? == [3, 5, 8, 13]);
    println!("safe-mem-own-identity: returned Value::MemOwn with intact contents");

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
