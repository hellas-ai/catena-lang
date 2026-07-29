use super::*;

const TILED_MATMUL_SOURCE: &str = include_str!("gpu/tiled_matmul.hex");

#[test]
#[ignore = "gpu.global/shared and gpu.launch do not have codegen yet"]
fn tiled_matmul_launch_allocates_output_before_launch() -> anyhow::Result<()> {
    let runtime = runtime_with(TILED_MATMUL_SOURCE)?;

    const A: usize = 16;
    const B: usize = 16;
    const C: usize = 16;

    let left_values = vec![1.0_f32; A * B];
    let right_values = vec![1.0_f32; B * C];
    let a = runtime.mem_f32(&left_values)?;
    let b = runtime.mem_f32(&right_values)?;

    let [result] = runtime.exec(
        "tiled-matmul-via-mem",
        [
            a,
            b,
            (A as u64).into(),
            (B as u64).into(),
            (C as u64).into(),
            ((A * B) as u64).into(),
            ((B * C) as u64).into(),
            ((A * C) as u64).into(),
            1_u64.into(),  // grid rows
            1_u64.into(),  // grid cols
            16_u64.into(), // block rows
            16_u64.into(), // block cols
        ],
    )?;
    let Value::Mem(result) = result else {
        anyhow::bail!("tiled-matmul-via-mem returned non-mem value: {result:?}");
    };

    assert_eq!(result.to_f32_vec(), vec![B as f32; A * C]);
    Ok(())
}
