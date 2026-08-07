use super::*;

const MATMUL_SOURCE: &str = include_str!("matrix/matmul.hex");
const MATMUL_BF16_SOURCE: &str = include_str!("matrix/matmul-bf16.hex");

#[test]
fn f32_matmul_row_major_bufs_from_mems() -> anyhow::Result<()> {
    let runtime = runtime_with(MATMUL_SOURCE)?;

    let a = runtime.mem_f32(&[
        1.0, 2.0, 3.0, //
        4.0, 5.0, 6.0,
    ])?;
    let b = runtime.mem_f32(&[
        7.0, 8.0, //
        9.0, 10.0, //
        11.0, 12.0,
    ])?;

    let [result] = runtime.exec(
        "matmul-2x2-via-mem",
        [
            a.as_ref().into(),
            b.as_ref().into(),
            2_u64.into(),
            2_u64.into(),
            3_u64.into(),
            6_u64.into(),
            6_u64.into(),
            4_u64.into(),
        ],
    )?;
    let Value::MemOwn(result) = result else {
        anyhow::bail!("matmul-2x2-via-mem returned non-mem value: {result:?}");
    };

    let values = result.to_f32_vec();
    let expected = [58.0_f32, 64.0, 139.0, 154.0];
    assert_eq!(values.len(), expected.len());
    for (actual, expected) in values.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() < 1e-4,
            "matmul output {actual} differed from expected {expected}"
        );
    }
    Ok(())
}

#[test]
fn f32_matmul_right_identity_view() -> anyhow::Result<()> {
    let runtime = runtime_with(MATMUL_SOURCE)?;

    let input_values = [1.0_f32, 2.0, 3.0, 4.0];
    let input = runtime.mem_f32(&input_values)?;
    let [result] = runtime.exec(
        "matmul-right-identity-2x2-via-mem",
        [input.as_ref().into(), 2_u64.into(), 4_u64.into()],
    )?;
    let Value::MemOwn(result) = result else {
        anyhow::bail!("matmul-right-identity-2x2-via-mem returned non-mem value: {result:?}");
    };

    assert_eq!(result.to_f32_vec(), input_values);
    Ok(())
}

#[test]
fn f32_matmul_left_identity_view() -> anyhow::Result<()> {
    let runtime = runtime_with(MATMUL_SOURCE)?;

    let input_values = [1.0_f32, 2.0, 3.0, 4.0];
    let input = runtime.mem_f32(&input_values)?;
    let [result] = runtime.exec(
        "matmul-left-identity-2x2-via-mem",
        [input.as_ref().into(), 2_u64.into(), 4_u64.into()],
    )?;
    let Value::MemOwn(result) = result else {
        anyhow::bail!("matmul-left-identity-2x2-via-mem returned non-mem value: {result:?}");
    };

    assert_eq!(result.to_f32_vec(), input_values);
    Ok(())
}

#[test]
fn bf16_matmul_row_major_bufs_from_mems() -> anyhow::Result<()> {
    use half::bf16;

    let runtime = runtime_with(MATMUL_BF16_SOURCE)?;
    let values = |values: &[f32]| {
        values
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>()
    };
    let a = runtime.mem_bf16(&values(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))?;
    let b = runtime.mem_bf16(&values(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]))?;

    let [result] = runtime.exec(
        "matmul-2x2-via-mem",
        [
            a.as_ref().into(),
            b.as_ref().into(),
            2_u64.into(),
            2_u64.into(),
            3_u64.into(),
            6_u64.into(),
            6_u64.into(),
            4_u64.into(),
        ],
    )?;
    let Value::MemOwn(result) = result else {
        anyhow::bail!("matmul-2x2-via-mem returned non-mem value: {result:?}");
    };

    assert_eq!(result.to_bf16_vec(), values(&[58.0, 64.0, 139.0, 154.0]));
    Ok(())
}

#[test]
fn bf16_matmul_right_identity_view() -> anyhow::Result<()> {
    use half::bf16;

    let runtime = runtime_with(MATMUL_BF16_SOURCE)?;
    let input_values = [1.0_f32, 2.0, 3.0, 4.0].map(bf16::from_f32);
    let input = runtime.mem_bf16(&input_values)?;
    let [result] = runtime.exec(
        "matmul-right-identity-2x2-via-mem",
        [input.as_ref().into(), 2_u64.into(), 4_u64.into()],
    )?;
    let Value::MemOwn(result) = result else {
        anyhow::bail!("matmul-right-identity-2x2-via-mem returned non-mem value: {result:?}");
    };

    assert_eq!(result.to_bf16_vec(), input_values);
    Ok(())
}

#[test]
fn bf16_matmul_left_identity_view() -> anyhow::Result<()> {
    use half::bf16;

    let runtime = runtime_with(MATMUL_BF16_SOURCE)?;
    let input_values = [1.0_f32, 2.0, 3.0, 4.0].map(bf16::from_f32);
    let input = runtime.mem_bf16(&input_values)?;
    let [result] = runtime.exec(
        "matmul-left-identity-2x2-via-mem",
        [input.as_ref().into(), 2_u64.into(), 4_u64.into()],
    )?;
    let Value::MemOwn(result) = result else {
        anyhow::bail!("matmul-left-identity-2x2-via-mem returned non-mem value: {result:?}");
    };

    assert_eq!(result.to_bf16_vec(), input_values);
    Ok(())
}
