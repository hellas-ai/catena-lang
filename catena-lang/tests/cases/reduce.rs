use super::*;

const BASIC_SOURCE: &str = include_str!("reduce/basic.hex");

#[test]
fn sum_empty_u64_reduce_exec() -> anyhow::Result<()> {
    let runtime = runtime_with(BASIC_SOURCE)?;

    let [result] = runtime.exec("sum-empty-u64-reduce", [])?;
    let Value::U64(result) = result else {
        anyhow::bail!("sum-empty-u64-reduce returned non-u64 value: {result:?}");
    };

    assert_eq!(result, 0);
    Ok(())
}
