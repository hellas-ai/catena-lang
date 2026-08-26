use super::*;

const FILL_WITH_READABLE_PAIR_SUM: &str = include_str!("launch/fill_with_readable_pair_sum.hex");

#[test]
fn perfect_tiling_fills_with_readable_pair_sum() -> anyhow::Result<()> {
    check_readable_pair_sum(8, 2, 4)
}

#[test]
fn predicated_tiling_fills_with_readable_pair_sum() -> anyhow::Result<()> {
    check_readable_pair_sum(7, 2, 4)
}

fn check_readable_pair_sum(
    destination_size: usize,
    grid_x: u64,
    block_x: u64,
) -> anyhow::Result<()> {
    let (runtime, artifact) = runtime_with(FILL_WITH_READABLE_PAIR_SUM)?;
    let destination = runtime.mem_u64(&vec![u64::MAX; destination_size])?;
    let source_values = [17_u64, 25, 99];
    let source = runtime.mem_u64(&source_values)?;

    let [destination, source] = runtime.exec(
        &artifact,
        "fill-with-readable-pair-sum",
        [
            destination.into(),
            source.into(),
            grid_x.into(),
            block_x.into(),
        ],
    )?;
    let Value::MemOwn(destination) = destination else {
        anyhow::bail!("fill-with-readable-pair-sum returned a non-memory destination")
    };
    let Value::MemOwn(source) = source else {
        anyhow::bail!("fill-with-readable-pair-sum returned a non-memory source")
    };

    assert_eq!(destination.to_u64_vec()?, vec![42; destination_size]);
    assert_eq!(source.to_u64_vec()?, source_values);
    Ok(())
}
