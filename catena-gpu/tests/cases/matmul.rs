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
    grid_x: u64,
    grid_y: u64,
    block_x: u64,
    block_y: u64,
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

    let [c, a, b] = runtime.exec(
        &artifact,
        "naive-u64-matmul",
        [
            c.into(),
            a.into(),
            b.into(),
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
    let Value::MemOwn(a) = a else {
        anyhow::bail!("naive-u64-matmul returned a non-memory A buffer")
    };
    let Value::MemOwn(b) = b else {
        anyhow::bail!("naive-u64-matmul returned a non-memory B buffer")
    };

    assert_eq!(c.to_u64_vec()?, [58, 64, 139, 154]);
    assert_eq!(a.to_u64_vec()?, a_values);
    assert_eq!(b.to_u64_vec()?, b_values);
    Ok(())
}
