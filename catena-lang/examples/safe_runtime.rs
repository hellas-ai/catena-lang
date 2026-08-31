use std::env;

use catena_lang::{
    codegen::GpuDialect,
    runtime::{MemOwn, Value, ValueKind},
    safe_runtime::{SafeExecError, SafeRuntime, run_safe_runtime_child_if_requested},
    stdlib,
};

const GPU_DIALECT_ENV: &str = "CATENA_GPU_DIALECT";
const OWNED_SOURCE: &str =
    "(def program mem-own-identity : (cap.own mem) -> (cap.own mem) = [memory])";
const ADD_ONE_SOURCE: &str =
    "(def program add-one : (u64 val) -> (u64 val) = ({_ u64.one} u64.add))";

fn main() -> anyhow::Result<()> {
    // If child mode is enabled after being spawned by the parent, run the child loop.
    // Otherwise, continue with the normal runtime execution as a parent process.
    if run_safe_runtime_child_if_requested()? {
        return Ok(());
    }

    let dialect = configured_gpu_dialect()?;
    let runtime = SafeRuntime::new(dialect)?;
    let artifact = runtime.load_sources(stdlib::sources().chain([
        include_str!("example.hex"),
        include_str!("materializec.hex"),
        OWNED_SOURCE,
    ]))?;
    let add_one = runtime.load_sources(stdlib::sources().chain([ADD_ONE_SOURCE]))?;
    let signature = add_one
        .entry_points()
        .iter()
        .find(|entry| entry.name() == "add-one")
        .ok_or_else(|| anyhow::anyhow!("add-one entry point is missing"))?;
    anyhow::ensure!(signature.inputs() == [ValueKind::U64]);
    anyhow::ensure!(signature.outputs() == [ValueKind::U64]);

    let [added] = runtime.exec(&add_one, "add-one", [41_u64.into()])?;
    anyhow::ensure!(matches!(added, Value::U64(42)));

    let [result] = runtime.exec(&artifact, "two-times-two", [])?;
    let Value::U64(result) = result else {
        anyhow::bail!("two-times-two returned non-u64 value: {result:?}");
    };
    println!("two-times-two: {result}");
    anyhow::ensure!(result == 4, "two-times-two returned {result}, expected 4");

    let [] = runtime.exec(&artifact, "require-true", [true.into()])?;

    let input = MemOwn::from_u64_slice(&[17, 19, 23], dialect)?;
    let [head] = runtime.exec(&artifact, "array-head-u64", [input.as_ref().into()])?;
    let Value::U64(head) = head else {
        anyhow::bail!("array-head-u64 returned non-u64 value: {head:?}");
    };
    anyhow::ensure!(head == 17, "array-head-u64 returned {head}, expected 17");
    let [head_again] = runtime.exec(&artifact, "array-head-u64", [input.as_ref().into()])?;
    anyhow::ensure!(matches!(head_again, Value::U64(17)));
    // The child only imported a borrowed mapping, so the parent allocation is
    // still valid after successive calls.
    anyhow::ensure!(input.try_to_u64_vec()? == [17, 19, 23]);
    println!("array-head-u64 through IPC: {head}");

    let [returned] = runtime.exec(&artifact, "mem-own-identity", [input.into()])?;
    let Value::MemOwn(returned) = returned else {
        anyhow::bail!("mem-own-identity returned non-owned memory: {returned:?}");
    };
    anyhow::ensure!(returned.try_to_u64_vec()? == [17, 19, 23]);

    let [materialized] = runtime.exec(&artifact, "materialize-indexes", [4_u64.into()])?;
    let Value::MemOwn(materialized) = materialized else {
        anyhow::bail!("materialize-indexes returned non-owned memory: {materialized:?}");
    };
    anyhow::ensure!(materialized.try_to_u64_vec()? == [1, 1, 1, 1]);

    let empty = MemOwn::from_u64_slice(&[], dialect)?;
    let [empty] = runtime.exec(&artifact, "mem-own-identity", [empty.into()])?;
    let Value::MemOwn(empty) = empty else {
        anyhow::bail!("empty mem-own-identity returned non-owned memory: {empty:?}");
    };
    anyhow::ensure!(empty.try_to_u64_vec()?.is_empty());

    match runtime.exec::<1, 0>(&artifact, "require-true", [false.into()]) {
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
    anyhow::ensure!(returned.try_to_u64_vec()? == [17, 19, 23]);
    anyhow::ensure!(materialized.try_to_u64_vec()? == [1, 1, 1, 1]);
    anyhow::ensure!(empty.try_to_u64_vec()?.is_empty());
    println!("owned outputs remained valid after child termination");

    // A device-side assertion is reported by the runtime synchronization
    // boundary. It must terminate and poison the isolated worker just like a
    // native host assertion, rather than leave a failed GPU context reusable.
    let mut gpu_fault_runtime = SafeRuntime::new(dialect)?;
    let gpu_fault_artifact = gpu_fault_runtime.load_sources(
        stdlib::sources().chain([include_str!("materializec.hex"), ADD_ONE_SOURCE]),
    )?;
    let fault_input = MemOwn::from_f32_slice(&[0.0; 8], dialect)?;
    let gate = MemOwn::from_u16_slice(&[0; 16], dialect)?;
    let up = MemOwn::from_u16_slice(&[0; 16], dialect)?;
    // The two weight tensors describe experts 0 and 1, so expert 2 triggers
    // the routed kernel's device assertion.
    let invalid_selected = MemOwn::from_u64_slice(&[2, 0], dialect)?;
    match gpu_fault_runtime.exec::<4, 4>(
        &gpu_fault_artifact,
        "materialize-borrow-routed-bf16-gemv-pair",
        [
            fault_input.into(),
            gate.as_ref().into(),
            up.as_ref().into(),
            invalid_selected.into(),
        ],
    ) {
        Err(SafeExecError::ChildTerminated { status, .. }) => {
            anyhow::ensure!(!status.success(), "GPU-faulting child exited successfully");
        }
        Err(error) => anyhow::bail!("device assertion returned the wrong error: {error}"),
        Ok(_) => anyhow::bail!("device assertion unexpectedly returned outputs"),
    }
    anyhow::ensure!(matches!(
        gpu_fault_runtime.exec::<1, 1>(&gpu_fault_artifact, "add-one", [41_u64.into()]),
        Err(SafeExecError::Unavailable { .. })
    ));
    println!("device assertion: child terminated and remained unavailable");

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
