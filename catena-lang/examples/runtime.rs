use std::path::PathBuf;

use catena_lang::runtime::{Runtime, Value};

fn main() -> anyhow::Result<()> {
    // Load standard library
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let runtime = Runtime::new([
        root.join("stdlib/cmc.hex"),
        root.join("stdlib/value.hex"),
        root.join("stdlib/buf.hex"),
        root.join("stdlib/index.hex"),
        root.join("stdlib/base/data.hex"),
        root.join("stdlib/base/fn.hex"),
        root.join("stdlib/base/product.hex"),
        root.join("stdlib/gpu.hex"),
        root.join("example.hex"),
    ])?;

    // Execute `not` from the source program with input value `false`
    let [result] = runtime.exec("not", [false.into()])?;
    println!("not(false): {result:?}");

    // U64 arithmetic
    let [two_times_two] = runtime.exec("two-times-two", [])?;
    let Value::U64(two_times_two) = two_times_two else {
        anyhow::bail!("two-times-two returned non-u64 value: {two_times_two:?}");
    };
    println!("two-times-two: {two_times_two}");
    anyhow::ensure!(
        two_times_two == 4,
        "two-times-two mismatch: got {two_times_two}, expected 4"
    );

    // F32 arithmetic
    let [arith_f32] = runtime.exec("arith-f32", [])?;
    let Value::F32(arith_f32) = arith_f32 else {
        anyhow::bail!("arith_f32 returned non-f32 value: {arith_f32:?}");
    };
    println!("arith_f32: {arith_f32}");
    anyhow::ensure!(
        arith_f32 == 4.0,
        "arith_f32 mismatch: got {arith_f32}, expected 4.0"
    );

    let [deadbeef] = runtime.exec("deadbeef", [])?;
    let Value::U64(deadbeef) = deadbeef else {
        anyhow::bail!("deadbeef returned non-u64 value: {deadbeef:?}");
    };
    println!("deadbeef: 0x{deadbeef:x}");
    anyhow::ensure!(
        deadbeef == 0xDEADBEEFDEADBEEF_u64,
        "deadbeef mismatch: got 0x{deadbeef:x}, expected 0xDEADBEEFDEADBEEF"
    );

    let [deadbeef32] = runtime.exec("deadbeef32", [])?;
    let Value::U32(deadbeef32) = deadbeef32 else {
        anyhow::bail!("deadbeef32 returned non-u32 value: {deadbeef32:?}");
    };
    println!("deadbeef32: 0x{deadbeef32:x}");
    anyhow::ensure!(
        deadbeef32 == 0xDEADBEEF_u32,
        "deadbeef32 mismatch: got 0x{deadbeef32:x}, expected 0xDEADBEEF"
    );

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
