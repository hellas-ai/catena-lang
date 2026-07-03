use super::*;

const BASIC_SOURCE: &str = include_str!("reducec/basic.hex");
const NAMED_PRIMITIVE_SOURCE: &str = include_str!("reducec/named_primitive.hex");
const SUM_SOURCE: &str = include_str!("reducec/sum.hex");

#[test]
fn sum_u64_exec() -> anyhow::Result<()> {
    let runtime = runtime_with(BASIC_SOURCE)?;

    let input = runtime.mem_u64(&[2, 3, 5, 7, 11])?;
    let [result] = runtime.exec("sum-u64", [input])?;
    let Value::U64(result) = result else {
        anyhow::bail!("sum-u64 returned non-u64 value: {result:?}");
    };

    assert_eq!(result, 28);
    Ok(())
}

#[test]
#[ignore = "depends on symbol resolution bug"]
fn reduce_with_named_primitive_exec() -> anyhow::Result<()> {
    let runtime = runtime_with(NAMED_PRIMITIVE_SOURCE)?;

    let [result] = runtime.exec("reduce-f32-fma-primitive", [1_u64.into()])?;
    let Value::F32(result) = result else {
        anyhow::bail!("reduce-f32-fma-primitive returned non-f32 value: {result:?}");
    };

    assert_eq!(result, 2.0);
    Ok(())
}

#[test]
fn dot_u64_exec() -> anyhow::Result<()> {
    let runtime = runtime_with(BASIC_SOURCE)?;

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

#[test]
fn sum_f32_exec() -> anyhow::Result<()> {
    let runtime = runtime_with(SUM_SOURCE)?;

    let input = runtime.mem_f32(&[1.5_f32, -0.5, 2.0, 4.0])?;
    let [result] = runtime.exec("sum-f32", [input])?;
    let Value::F32(result) = result else {
        anyhow::bail!("sum-f32 returned non-f32 value: {result:?}");
    };

    assert_eq!(result, 7.0);
    Ok(())
}

#[test]
fn mean_f32_exec() -> anyhow::Result<()> {
    let runtime = runtime_with(SUM_SOURCE)?;

    let input = runtime.mem_f32(&[1.5_f32, -0.5, 2.0, 4.0])?;
    let [result] = runtime.exec("mean-f32", [input])?;
    let Value::F32(result) = result else {
        anyhow::bail!("mean-f32 returned non-f32 value: {result:?}");
    };

    assert_eq!(result, 1.75);
    Ok(())
}

#[test]
fn max_f32_exec() -> anyhow::Result<()> {
    let runtime = runtime_with(SUM_SOURCE)?;

    let input = runtime.mem_f32(&[1.5_f32, -0.5, 2.0, 4.0])?;
    let [result] = runtime.exec("max-f32", [input])?;
    let Value::F32(result) = result else {
        anyhow::bail!("max-f32 returned non-f32 value: {result:?}");
    };

    assert_eq!(result, 4.0);
    Ok(())
}
