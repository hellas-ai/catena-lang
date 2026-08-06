//! SafeRuntime memory-ownership regressions.
//!
//! This is a custom-harness test because SafeRuntime respawns the current
//! executable. `main` must enter child mode before a Rust test harness would
//! parse arguments or run tests.

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

(def program safe-two-owned-identity :
  {(cap.own mem) (cap.own mem)}
  ->
  {(cap.own mem) (cap.own mem)}
= ([left right.] [.left right])
)

(def program safe-add-first-and-return-owned :
  {(cap.own mem) (cap.ref mem)}
  ->
  {(cap.own mem) (u64 val)}
= ([owned-mem borrowed-mem.]
  {
    ([.owned-mem] mem.cast.u64 [owned-len owned-buffer.])
    ([.borrowed-mem] mem.cast.u64 [borrowed-len borrowed-buffer.])
    ([.owned-len] u64.assert-nz [owned-nonempty.])
    ([.borrowed-len] u64.assert-nz [borrowed-nonempty.])
    ([.owned-nonempty] ix.zero [owned-zero.])
    ([.borrowed-nonempty] ix.zero [borrowed-zero.])
    ([.owned-zero owned-buffer] ix [owned-first.])
    ([.borrowed-zero borrowed-buffer] ix [borrowed-first.])
    ([.owned-first borrowed-first] u64.add [sum.])
    ([.owned-len owned-buffer] buf.to-mem [returned-owned.])
  }
  [.returned-owned sum]
))
"#;

fn main() -> anyhow::Result<()> {
    if run_safe_runtime_child_if_requested()? {
        return Ok(());
    }

    let runtime = SafeRuntime::from_sources(
        stdlib::sources().chain([include_str!("../examples/example.hex"), MEMORY_SOURCE]),
        configured_gpu_dialect()?,
    )?;

    scalar_execution(&runtime)?;
    owned_identity(&runtime)?;
    borrowed_input_is_reusable(&runtime)?;
    mixed_transfer_and_borrow(&runtime)?;
    multiple_owned_outputs(&runtime)?;
    child_crash_is_isolated(&runtime)?;
    Ok(())
}

fn scalar_execution(runtime: &SafeRuntime) -> anyhow::Result<()> {
    let [result] = runtime.exec("two-times-two", [])?;
    anyhow::ensure!(matches!(result, Value::U64(4)));
    Ok(())
}

fn owned_identity(runtime: &SafeRuntime) -> anyhow::Result<()> {
    let expected = [3_u64, 5, 8, 13];
    let input = runtime.mem_u64(&expected)?;
    let [output] = runtime.exec("safe-mem-own-identity", [input.into()])?;
    let Value::MemOwn(output) = output else {
        anyhow::bail!("owned identity returned non-owned memory: {output:?}");
    };
    anyhow::ensure!(output.try_to_u64_vec()? == expected);
    Ok(())
}

fn borrowed_input_is_reusable(runtime: &SafeRuntime) -> anyhow::Result<()> {
    let input = runtime.mem_u64(&[17, 19, 23])?;
    for _ in 0..2 {
        let [head] = runtime.exec("array-head-u64", [input.as_ref().into()])?;
        anyhow::ensure!(matches!(head, Value::U64(17)));
    }
    anyhow::ensure!(input.try_to_u64_vec()? == [17, 19, 23]);
    Ok(())
}

fn mixed_transfer_and_borrow(runtime: &SafeRuntime) -> anyhow::Result<()> {
    let owned = runtime.mem_u64(&[3, 5])?;
    let borrowed = runtime.mem_u64(&[8, 13])?;
    let [returned, sum] = runtime.exec(
        "safe-add-first-and-return-owned",
        [owned.into(), borrowed.as_ref().into()],
    )?;
    let Value::MemOwn(returned) = returned else {
        anyhow::bail!("mixed program returned non-owned memory: {returned:?}");
    };
    anyhow::ensure!(matches!(sum, Value::U64(11)));
    anyhow::ensure!(returned.try_to_u64_vec()? == [3, 5]);
    anyhow::ensure!(borrowed.try_to_u64_vec()? == [8, 13]);
    Ok(())
}

fn multiple_owned_outputs(runtime: &SafeRuntime) -> anyhow::Result<()> {
    let left = runtime.mem_u64(&[29, 31])?;
    let right = runtime.mem_u64(&[37, 41])?;
    let [left, right] = runtime.exec("safe-two-owned-identity", [left.into(), right.into()])?;
    let Value::MemOwn(left) = left else {
        anyhow::bail!("first output was not owned memory: {left:?}");
    };
    let Value::MemOwn(right) = right else {
        anyhow::bail!("second output was not owned memory: {right:?}");
    };
    anyhow::ensure!(left.try_to_u64_vec()? == [29, 31]);
    anyhow::ensure!(right.try_to_u64_vec()? == [37, 41]);
    Ok(())
}

fn child_crash_is_isolated(runtime: &SafeRuntime) -> anyhow::Result<()> {
    match runtime.exec::<1, 0>("require-true", [false.into()]) {
        Err(SafeExecError::ChildTerminated { status, stderr }) => {
            anyhow::ensure!(!status.success());
            anyhow::ensure!(stderr.contains("catena assertion failed"));
            Ok(())
        }
        Err(error) => anyhow::bail!("assertion returned the wrong error: {error}"),
        Ok([]) => anyhow::bail!("assertion unexpectedly succeeded"),
    }
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
