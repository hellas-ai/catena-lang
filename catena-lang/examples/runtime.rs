use std::{env, path::PathBuf};

use catena_lang::{
    codegen::GpuDialect,
    runtime::{Runtime, Value},
    stdlib,
};

const GPU_DIALECT_ENV: &str = "CATENA_GPU_DIALECT";
const ARRAY_HEAD_PLUS_ONE: &str = r#"
(def program array-head-u64 : ([n.] (cap.ref mem)) -> ([n.] (u64 val)) = (
  mem.cast.u64
  {
    (u64.assert-nz ix.zero)
    [b]
  }
  ix
  ({_ u64.one} u64.add)
))
"#;

fn main() -> anyhow::Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut runtime = Runtime::new(configured_gpu_dialect()?)?;
    let artifact =
        runtime.load(stdlib::paths_from(&root).chain([root.join("examples/example.hex")]))?;
    let plus_one = runtime.load_sources(stdlib::sources().chain([ARRAY_HEAD_PLUS_ONE]))?;

    let [result] = runtime.exec(&artifact, "two-times-two", [])?;
    let Value::U64(result) = result else {
        anyhow::bail!("two-times-two returned non-u64 value: {result:?}");
    };
    println!("two-times-two: {result}");
    anyhow::ensure!(result == 4, "two-times-two returned {result}, expected 4");

    let [] = runtime.exec(&artifact, "require-true", [true.into()])?;

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
    let [head] = runtime.exec(&artifact, "array-head-u64", [input.as_ref().into()])?;
    let Value::U64(head) = head else {
        anyhow::bail!("array-head-u64 returned non-u64 value: {head:?}");
    };

    println!("array-head-u64: 0x{head:x} (expected 0x{:x})", values[0]);
    anyhow::ensure!(
        head == values[0],
        "array head mismatch: got 0x{head:x}, expected 0x{:x}",
        values[0]
    );

    // Run the second .so's version of the same program with the device
    // allocation created above.
    let [head_plus_one] = runtime.exec(&plus_one, "array-head-u64", [input.as_ref().into()])?;
    let Value::U64(head_plus_one) = head_plus_one else {
        anyhow::bail!("second array-head-u64 returned non-u64 value: {head_plus_one:?}");
    };
    anyhow::ensure!(head_plus_one == values[0] + 1);
    anyhow::ensure!(input.try_to_u64_vec()? == values);
    println!("same allocation through second artifact: 0x{head_plus_one:x}");

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
