use super::*;

use std::os::unix::process::ExitStatusExt;
use std::process::Command;

use catena_lang::runtime::Value;

const SHARED_MEMORY_LIMIT_SOURCE: &str = include_str!("parallel/shared_memory_limit.hex");
const SHARED_MEMORY_LIMIT_CHILD: &str = "CATENA_SHARED_MEMORY_LIMIT_CHILD";
const SHARED_MEMORY_LIMIT_HELPER: &str =
    "cases::parallel::shared_memory_limit_aborts_in_isolated_child";

// This is intentionally excluded from the ordinary runtime suite: it expects
// generated native code to abort and may monopolize the GPU while doing so.
// Run it alone with:
//
//   cargo test -p catena-lang --features runtime-tests --test runtime \
//     cases::parallel::excessive_shared_memory_aborts_execution \
//     -- --exact --ignored --nocapture --test-threads=1
#[test]
#[ignore = "GPU death test; run explicitly in isolation using the command above"]
fn excessive_shared_memory_aborts_execution() -> anyhow::Result<()> {
    // A GPU resource failure aborts generated native code. Run only the helper in
    // a child test process so the rest of this test binary remains unaffected.
    let output = Command::new(std::env::current_exe()?)
        .args([
            SHARED_MEMORY_LIMIT_HELPER,
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(SHARED_MEMORY_LIMIT_CHILD, "1")
        .output()?;

    assert_eq!(
        output.status.signal(),
        Some(libc::SIGABRT),
        "excessive shared memory did not abort: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("catena GPU launch error:"),
        "child failed without the generated GPU diagnostic: {stderr}"
    );
    Ok(())
}

#[test]
#[ignore = "death-test helper; run only through excessive_shared_memory_aborts_execution"]
fn shared_memory_limit_aborts_in_isolated_child() -> anyhow::Result<()> {
    if std::env::var_os(SHARED_MEMORY_LIMIT_CHILD).is_none() {
        return Ok(());
    }

    let runtime = runtime_with(SHARED_MEMORY_LIMIT_SOURCE)?;
    // Establish that this compiler/runtime/device can launch the same kernel;
    // otherwise an unavailable GPU could masquerade as the expected failure.
    let [valid] = runtime.exec("launch-with-shared-elements", [0_u64.into(), 1_u64.into()])?;
    let Value::MemOwn(valid) = valid else {
        anyhow::bail!("valid control launch returned a non-memory value")
    };
    drop(valid);

    // 2^28 f32 elements request one GiB of dynamic shared memory, deliberately
    // far beyond a per-block capacity. Avoid relying on a vendor error code.
    let _: [Value<'static>; 1] = runtime.exec(
        "launch-with-shared-elements",
        [(1_u64 << 28).into(), 1_u64.into()],
    )?;
    anyhow::bail!("excessive shared-memory launch unexpectedly returned")
}
