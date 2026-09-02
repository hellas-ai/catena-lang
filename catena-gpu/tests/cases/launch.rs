use super::*;

const FILL_WITH_READABLE_SUM: &str = include_str!("launch/fill_with_readable_sum.hex");

#[test]
fn perfect_tiling_fills_with_readable_sum() -> anyhow::Result<()> {
    check_readable_sum(8, 2, 4)
}

#[test]
fn predicated_tiling_fills_with_readable_sum() -> anyhow::Result<()> {
    check_readable_sum(7, 2, 4)
}

fn check_readable_sum(destination_size: usize, grid_x: u32, block_x: u32) -> anyhow::Result<()> {
    let (runtime, artifact) = runtime_with(FILL_WITH_READABLE_SUM)?;
    let destination = runtime.mem_u64(&vec![u64::MAX; destination_size])?;
    let source_values = [17_u64, 25, 99];
    let source = runtime.mem_u64(&source_values)?;

    let [destination] = artifact.exec(
        "fill-with-readable-sum",
        [
            destination.into(),
            source.as_ref().into(),
            grid_x.into(),
            block_x.into(),
        ],
    )?;
    let Value::MemOwn(destination) = destination else {
        anyhow::bail!("fill-with-readable-sum returned a non-memory destination")
    };
    assert_eq!(destination.try_to_u64_vec()?, vec![141; destination_size]);
    assert_eq!(source.try_to_u64_vec()?, source_values);
    Ok(())
}
