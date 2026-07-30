use super::*;

const TILED_MATMUL_SOURCE: &str = include_str!("gpu/tiled_matmul.hex");

#[test]
fn tiled_matmul_launch_allocates_output_before_launch() -> anyhow::Result<()> {
    let runtime = runtime_with(TILED_MATMUL_SOURCE)?;

    const ROWS: usize = 16;
    const INNER: usize = 16;
    const COLS: usize = 16;

    let left_values = vec![1.0_f32; ROWS * INNER];
    let right_values = vec![1.0_f32; INNER * COLS];
    let a = runtime.mem_f32(&left_values)?;
    let b = runtime.mem_f32(&right_values)?;

    let [result] = runtime.exec(
        "tiled-matmul-via-mem",
        [
            a,
            b,
            (ROWS as u64).into(),
            (COLS as u64).into(),
            (INNER as u64).into(),
            16_u64.into(),
        ],
    )?;
    let Value::Mem(result) = result else {
        anyhow::bail!("tiled-matmul-via-mem returned non-mem value: {result:?}");
    };

    assert_eq!(result.to_f32_vec(), vec![INNER as f32; ROWS * COLS]);
    Ok(())
}
