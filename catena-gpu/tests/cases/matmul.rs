use super::*;

const NAIVE_U64: &str = include_str!("matmul/naive_u64.hex");

#[test]
fn perfect_tiling_computes_naive_u64_matmul() -> anyhow::Result<()> {
    check_naive_u64_matmul(1, 1, 2, 2)
}

#[test]
fn predicated_tiling_computes_naive_u64_matmul() -> anyhow::Result<()> {
    check_naive_u64_matmul(1, 1, 3, 2)
}

fn check_naive_u64_matmul(
    grid_x: u32,
    grid_y: u32,
    block_x: u32,
    block_y: u32,
) -> anyhow::Result<()> {
    let (runtime, artifact) = runtime_with(NAIVE_U64)?;
    let c = runtime.mem_u64(&[u64::MAX; 4])?;
    let a_values = [
        1_u64, 2, 3, //
        4, 5, 6,
    ];
    let b_values = [
        7_u64, 8, //
        9, 10, //
        11, 12,
    ];
    let a = runtime.mem_u64(&a_values)?;
    let b = runtime.mem_u64(&b_values)?;

    let [c] = artifact.exec(
        "naive-u64-matmul",
        [
            c.into(),
            a.as_ref().into(),
            b.as_ref().into(),
            2_u64.into(),
            3_u64.into(),
            2_u64.into(),
            grid_x.into(),
            grid_y.into(),
            block_x.into(),
            block_y.into(),
        ],
    )?;
    let Value::MemOwn(c) = c else {
        anyhow::bail!("naive-u64-matmul returned a non-memory C buffer")
    };

    assert_eq!(c.try_to_u64_vec()?, [58, 64, 139, 154]);
    assert_eq!(a.try_to_u64_vec()?, a_values);
    assert_eq!(b.try_to_u64_vec()?, b_values);
    Ok(())
}
