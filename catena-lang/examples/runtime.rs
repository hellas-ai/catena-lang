use std::env;
use std::path::PathBuf;

use catena_lang::{
    codegen::GpuDialect,
    runtime::{Runtime, Value},
    stdlib,
};

const GPU_DIALECT_ENV: &str = "CATENA_GPU_DIALECT";

fn main() -> anyhow::Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dialect = configured_gpu_dialect()?;
    let runtime = Runtime::new(
        stdlib::paths_from(&root).chain([root.join("examples/example.hex")]),
        dialect,
    )?;

    // Input values for `array-head-u64`
    let values = [0x123456789abcdef0_u64, 7, 11];
    println!(
        "array-head input: [{}]",
        values
            .iter()
            .map(|value| format!("0x{value:x}"))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Execute array-head-u64 with values above
    let input = runtime.mem_u64(&values)?;
    let [head] = runtime.exec("array-head-u64", [input])?;
    let Value::U64(head) = head else {
        anyhow::bail!("array-head-u64 returned non-u64 value: {head:?}");
    };

    println!("array-head-u64: 0x{head:x} (expected 0x{:x})", values[0]);
    anyhow::ensure!(
        head == values[0],
        "array head mismatch: got 0x{head:x}, expected 0x{:x}",
        values[0]
    );

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
