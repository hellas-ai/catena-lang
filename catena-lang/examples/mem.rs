//! Demonstrates how the runtime models device-memory ownership.
//!
//! `MemOwn::into()` transfers an allocation to the called Catena program, while
//! `MemOwn::as_ref().into()` passes a lifetime-bound `MemRef` without consuming
//! its owner. The program returns the transferred allocation as `cap.own mem`
//! alongside a scalar computed from both buffers. Consequently, the first input
//! cannot be used after the call, the second remains usable, and the returned
//! memory is observably `Value::MemOwn`.

use std::env;

use catena_lang::{
    codegen::GpuDialect,
    runtime::{Runtime, Value},
    stdlib,
};

const GPU_DIALECT_ENV: &str = "CATENA_GPU_DIALECT";

const SOURCE: &str = r#"
(def program add-first-and-return-owned :
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
    let mut runtime = Runtime::new(configured_gpu_dialect()?)?;
    runtime.load_sources(stdlib::sources().chain([SOURCE]))?;

    let owned = runtime.mem_u64(&[3, 5])?;
    let borrowed = runtime.mem_u64(&[8, 13])?;

    let [returned, sum] = runtime.exec(
        "add-first-and-return-owned",
        [owned.into(), borrowed.as_ref().into()],
    )?;

    let Value::MemOwn(returned) = returned else {
        anyhow::bail!("program returned non-owned memory: {returned:?}");
    };
    let Value::U64(sum) = sum else {
        anyhow::bail!("program returned a non-u64 sum: {sum:?}");
    };

    anyhow::ensure!(sum == 11, "expected 3 + 8 = 11, got {sum}");
    anyhow::ensure!(returned.try_to_u64_vec()? == [3, 5]);
    anyhow::ensure!(borrowed.try_to_u64_vec()? == [8, 13]);

    println!("returned Value::MemOwn and computed 3 + 8 = {sum}");
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
