use super::*;

const TILED_MATMUL_SOURCE: &str = include_str!("gpu/tiled_matmul.hex");

fn result_mem([result]: [Value; 1], program: &str) -> anyhow::Result<Vec<f32>> {
    let Value::Mem(result) = result else {
        anyhow::bail!("{program} returned non-mem value: {result:?}");
    };
    Ok(result.to_f32_vec())
}

fn reference_matmul(a: &[f32], b: &[f32], rows: usize, inner: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .flat_map(|row| {
            (0..cols).map(move |col| {
                (0..inner)
                    .map(|k| a[row * inner + k] * b[k * cols + col])
                    .sum()
            })
        })
        .collect()
}

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

    let result = runtime.exec(
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

    assert_eq!(
        result_mem(result, "tiled-matmul-via-mem")?,
        vec![INNER as f32; ROWS * COLS]
    );
    Ok(())
}

#[test]
fn tiled_matmul_multiple_output_and_inner_tiles() -> anyhow::Result<()> {
    let runtime = runtime_with(TILED_MATMUL_SOURCE)?;

    const ROWS: usize = 4;
    const INNER: usize = 6;
    const COLS: usize = 6;
    const TILE_SIZE: usize = 2;

    let a = (0..ROWS * INNER)
        .map(|index| (index % 7) as f32 - 3.0)
        .collect::<Vec<_>>();
    let b = (0..INNER * COLS)
        .map(|index| (index * 5 % 7) as f32 - 3.0)
        .collect::<Vec<_>>();
    let expected = reference_matmul(&a, &b, ROWS, INNER, COLS);

    let a_mem = runtime.mem_f32(&a)?;
    let b_mem = runtime.mem_f32(&b)?;
    let result = runtime.exec(
        "tiled-matmul-via-mem",
        [
            a_mem,
            b_mem,
            (ROWS as u64).into(),
            (COLS as u64).into(),
            (INNER as u64).into(),
            (TILE_SIZE as u64).into(),
        ],
    )?;

    assert_eq!(result_mem(result, "tiled-matmul-via-mem")?, expected);
    Ok(())
}

#[test]
fn tiled_matmul_right_identity_closure() -> anyhow::Result<()> {
    let runtime = runtime_with(TILED_MATMUL_SOURCE)?;
    const N: usize = 4;
    const TILE_SIZE: usize = 2;
    let values = (0..N * N)
        .map(|index| index as f32 - 5.0)
        .collect::<Vec<_>>();

    let input = runtime.mem_f32(&values)?;
    let result = runtime.exec(
        "tiled-matmul-right-identity-via-mem",
        [input, (N as u64).into(), (TILE_SIZE as u64).into()],
    )?;

    assert_eq!(
        result_mem(result, "tiled-matmul-right-identity-via-mem")?,
        values
    );
    Ok(())
}

#[test]
fn tiled_matmul_left_identity_closure() -> anyhow::Result<()> {
    let runtime = runtime_with(TILED_MATMUL_SOURCE)?;
    const N: usize = 4;
    const TILE_SIZE: usize = 2;
    let values = (0..N * N)
        .map(|index| 8.0 - index as f32)
        .collect::<Vec<_>>();

    let input = runtime.mem_f32(&values)?;
    let result = runtime.exec(
        "tiled-matmul-left-identity-via-mem",
        [input, (N as u64).into(), (TILE_SIZE as u64).into()],
    )?;

    assert_eq!(
        result_mem(result, "tiled-matmul-left-identity-via-mem")?,
        values
    );
    Ok(())
}
