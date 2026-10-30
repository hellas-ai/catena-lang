//! Minimal reproduction: owned device state returned by one `SafeRuntime`
//! call cannot be passed back into the next call on HIP.
//!
//! Observed on catena-lang 9ec3f625, TheRock HIP 7.15.0, Strix Halo gfx1151:
//! feeding the child-allocated 56 MiB state back in fails at step=0 with
//! `Hip runtime failed to export IPC memory handle with status 1`.
//! The same program with an 8-byte state passes.
//! Requires a HIP GPU: `cargo test -p catena-lang --features runtime-tests
//! --test safe_runtime_owned_state`. Expected to fail until fixed.
use catena_lang::runtime::{
    GeneratedFunction, GpuDialect, MemOwn, RuntimeModule, Value, ValueKind,
};
use catena_lang::safe_runtime::{SafeRuntime, run_safe_runtime_child_if_requested};

const HIP: GpuDialect = GpuDialect::Hip;
/// The state size from the original failing model trace (56 MiB).
const STATE_BYTES: u64 = 58_720_256;

/// `alloc(n)` returns fresh device memory; `step` hands its owned state back
/// unchanged and allocates a new token.
fn module() -> RuntimeModule {
    let source = r#"#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdlib>
struct Mem { void* data; uint64_t len; };
extern "C" void alloc(uint64_t n, Mem* out) { void* p=nullptr; if (n && hipMalloc(&p,n)!=hipSuccess) abort(); *out={p,n}; }
extern "C" void step(Mem token, Mem state, Mem* state_out, Mem* next) { *state_out=state; alloc(8,next); }
"#;
    RuntimeModule::new(
        HIP,
        source,
        [
            GeneratedFunction {
                source_name: "alloc".into(),
                symbol: "alloc".into(),
                inputs: vec![ValueKind::U64],
                outputs: vec![ValueKind::MemOwn],
            },
            GeneratedFunction {
                source_name: "step".into(),
                symbol: "step".into(),
                inputs: vec![ValueKind::MemRef, ValueKind::MemOwn],
                outputs: vec![ValueKind::MemOwn; 2],
            },
        ],
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if run_safe_runtime_child_if_requested()? {
        return Ok(());
    }
    let mut safe = SafeRuntime::new(HIP)?;
    let artifact = safe.load(module())?;
    // State created by, and returned from, the child.
    let mut state = artifact
        .exec_values("alloc", vec![STATE_BYTES.into()])?
        .pop()
        .unwrap();
    let mut token = MemOwn::from_u64_slice(&[41], HIP)?;
    for step in 0..2 {
        let mut out = artifact
            .exec_values("step", vec![token.as_ref().into(), state])
            .map_err(|e| format!("step={step}: {e}"))?;
        let Value::MemOwn(next) = out.pop().unwrap() else {
            unreachable!()
        };
        token = next;
        state = out.pop().unwrap();
    }
    println!("owned state round-tripped through two SafeRuntime calls");
    Ok(())
}
