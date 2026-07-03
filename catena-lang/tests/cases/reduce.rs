use super::*;

const BASIC_SOURCE: &str = include_str!("reduce/basic.hex");
const SUM_ONES_U64_SOURCE: &str = include_str!("reduce/sum_ones_u64.hex");

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

#[test]
fn sum_ones_u64_reduce_uses_input_length() -> anyhow::Result<()> {
    let runtime = runtime_with(SUM_ONES_U64_SOURCE)?;

    let [result] = runtime.exec("sum-ones-u64-reduce", [4_u64.into()])?;
    let Value::U64(result) = result else {
        anyhow::bail!("sum-ones-u64-reduce returned non-u64 value: {result:?}");
    };

    assert_eq!(result, 4);
    Ok(())
}
