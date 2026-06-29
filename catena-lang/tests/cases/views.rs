use super::*;

const ROW_MAJOR_SOURCE: &str = include_str!("views/row_major.hex");

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
