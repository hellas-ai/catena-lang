use super::*;

const SOURCE: &str = include_str!("../../examples/reducec.hex");

#[test]
fn sum_u64_exec() -> anyhow::Result<()> {
    let runtime = runtime_with(SOURCE)?;

    let input = runtime.mem_u64(&[2, 3, 5, 7, 11])?;
    let [result] = runtime.exec("sum-u64", [input])?;
    let Value::U64(result) = result else {
        anyhow::bail!("sum-u64 returned non-u64 value: {result:?}");
    };

    assert_eq!(result, 28);
    Ok(())
}

#[test]
#[ignore = "dot-u64 requires product environment lowering for reducec producers"]
fn dot_u64_exec() -> anyhow::Result<()> {
    let runtime = runtime_with(SOURCE)?;

    let lhs = runtime.mem_u64(&[2, 3, 5, 7])?;
    let rhs = runtime.mem_u64(&[11, 13, 17, 19])?;
    let [result] = runtime.exec("dot-u64", [lhs, rhs])?;
    let Value::U64(result) = result else {
        anyhow::bail!("dot-u64 returned non-u64 value: {result:?}");
    };

    let dot = 2 * 11 + 3 * 13 + 5 * 17 + 7 * 19;
    assert_eq!(result, dot);
    Ok(())
}
