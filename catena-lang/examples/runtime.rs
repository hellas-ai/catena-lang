use std::path::PathBuf;

use catena_lang::runtime::{Runtime, Value};

fn main() -> anyhow::Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let runtime = Runtime::new([
        root.join("stdlib/cmc.hex"),
        root.join("stdlib/value.hex"),
        root.join("stdlib/buf.hex"),
        root.join("stdlib/index.hex"),
        root.join("stdlib/data.hex"),
        root.join("stdlib/fn.hex"),
        root.join("stdlib/product.hex"),
        root.join("stdlib/gpu.hex"),
        root.join("examples/example.hex"),
        root.join("examples/sincos.hex"),
    ])?;

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

    let input = 1.0_f32;
    let [sin_approx_output] = runtime.exec("sin-approx", [input.into()])?;
    let Value::F32(sin_approx_output) = sin_approx_output else {
        anyhow::bail!("sin-approx returned non-f32 value: {sin_approx_output:?}");
    };

    let expected = input.sin();
    println!("sin-approx (x = 1.0): {sin_approx_output} (expected {expected})");
    anyhow::ensure!(
        (sin_approx_output - expected).abs() < 1e-4,
        "sin-approx output mismatch: got {sin_approx_output}, expected {expected}"
    );

    let input = 4.0_f32;
    let [full_output] = runtime.exec("sin-approx-full", [input.into()])?;
    let Value::F32(full_output) = full_output else {
        anyhow::bail!("sin-approx-full returned non-f32 value: {full_output:?}");
    };

    let expected = input.sin();
    println!("sin-approx-full (x = {input}): {full_output} (sin {expected})");
    anyhow::ensure!(
        (full_output - expected).abs() < 1e-5,
        "sin-approx-full output mismatch for x={input}: got {full_output}, expected {expected}"
    );

    Ok(())
}
