use super::*;

const ROW_MAJOR_SOURCE: &str = include_str!("views/row_major.hex");
const ROW_MAJOR_DIAGONAL_SUM_REDUCE_SOURCE: &str =
    include_str!("views/row_major_diagonal_sum_reduce.hex");

#[test]
#[ignore = "depends on closure materialization lowering"]
fn row_major_view_diagonal_materialize_exec() -> anyhow::Result<()> {
    let runtime = runtime_with(ROW_MAJOR_SOURCE)?;

    let input = runtime.mem_u64(&[10, 11, 12, 20, 21, 22, 30, 31, 32])?;
    let [result] = runtime.exec("matrix-diagonal-u64", [3_u64.into(), input])?;
    let Value::Mem(result) = result else {
        anyhow::bail!("matrix-diagonal-u64 returned non-mem value: {result:?}");
    };

    assert_eq!(result.to_u64_vec(), vec![10, 21, 32]);
    Ok(())
}

#[test]
#[ignore = "depends on monogamous closure conversion for buf.row-major-view"]
fn row_major_view_diagonal_reduce_exec() -> anyhow::Result<()> {
    let runtime = runtime_with(ROW_MAJOR_DIAGONAL_SUM_REDUCE_SOURCE)?;

    let input = runtime.mem_u64(&[10, 11, 12, 20, 21, 22, 30, 31, 32])?;
    let [result] = runtime.exec("matrix-diagonal-sum-u64", [3_u64.into(), input])?;
    let Value::U64(result) = result else {
        anyhow::bail!("matrix-diagonal-sum-u64 returned non-u64 value: {result:?}");
    };

    assert_eq!(result, 63);
    Ok(())
}
