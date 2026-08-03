use std::{env, path::PathBuf};

use catena_lang::{
    codegen::GpuDialect,
    runtime::Value,
    safe_runtime::{SafeExecError, SafeRuntime, run_safe_runtime_child_if_requested},
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
        stdlib::paths_from(&root).chain([
            root.join("examples/example.hex"),
            root.join("examples/materializec.hex"),
        ]),
        configured_gpu_dialect()?,
    )?;

    let [result] = runtime.exec("two-times-two", [])?;
    let Value::U64(result) = result else {
        anyhow::bail!("two-times-two returned non-u64 value: {result:?}");
    };
    println!("two-times-two: {result}");
    anyhow::ensure!(result == 4, "two-times-two returned {result}, expected 4");

    let [] = runtime.exec("require-true", [true.into()])?;

    let values = [0x123456789abcdef0_u64, 7, 11];
    let input = runtime.mem_u64(&values)?;
    let retained_input = input.clone();
    let [head] = runtime.exec("array-head-u64", [input])?;
    let Value::U64(head) = head else {
        anyhow::bail!("array-head-u64 returned non-u64 value: {head:?}");
    };
    println!("array-head-u64: 0x{head:x} (expected 0x{:x})", values[0]);
    anyhow::ensure!(head == values[0], "array head mismatch");

    let [materialized] = runtime.exec("materialize-indexes", [4_u64.into()])?;
    let Value::Mem(memory) = &materialized else {
        anyhow::bail!("materialize-indexes returned non-memory value: {materialized:?}");
    };
    anyhow::ensure!(memory.try_to_u64_vec()? == [1, 1, 1, 1]);

    let [head] = runtime.exec("array-head-u64", [materialized.clone()])?;
    let Value::U64(head) = head else {
        anyhow::bail!("array-head-u64 returned non-u64 value: {head:?}");
    };
    anyhow::ensure!(head == 1, "reused child memory had the wrong first value");
    println!("child memory result: read in parent and reused in child");
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

    let Value::Mem(retained_input) = retained_input else {
        unreachable!("mem_u64 must return a memory value");
    };
    anyhow::ensure!(retained_input.try_to_u64_vec()? == values);
    println!("parent device buffer remained valid after child termination");

    let Value::Mem(materialized) = materialized else {
        unreachable!("materialize-indexes must return a memory value");
    };
    anyhow::ensure!(materialized.try_to_u64_vec()? == [1, 1, 1, 1]);
    println!("copied child result remained valid after child termination");

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
