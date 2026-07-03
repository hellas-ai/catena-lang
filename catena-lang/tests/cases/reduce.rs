use super::*;

const BASIC_SOURCE: &str = include_str!("reduce/basic.hex");
const AMBIENT_SOURCE: &str = include_str!("reduce/ambient.hex");

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
#[ignore = "source-level reduce with ambient index variables is not runtime-clean yet"]
fn sum_ones_u64_reduce_uses_ambient_length() -> anyhow::Result<()> {
    let runtime = runtime_with(AMBIENT_SOURCE)?;

    let input = runtime.mem_u64(&[0, 0, 0, 0])?;
    let [result] = runtime.exec("sum-ones-u64-reduce", [input])?;
    let Value::U64(result) = result else {
        anyhow::bail!("sum-ones-u64-reduce returned non-u64 value: {result:?}");
    };

    assert_eq!(result, 4);
    Ok(())
}
