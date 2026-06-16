use catena_lang::{
    codegen::GpuDialect,
    runtime::{Runtime, Value},
};

const GPU_DIALECT_ENV: &str = "CATENA_GPU_DIALECT";

const STDLIB: &[&str] = &[
    include_str!("../stdlib/cmc.hex"),
    include_str!("../stdlib/value.hex"),
    include_str!("../stdlib/buf.hex"),
    include_str!("../stdlib/index.hex"),
    include_str!("../stdlib/data.hex"),
    include_str!("../stdlib/fn.hex"),
    include_str!("../stdlib/product.hex"),
    include_str!("../stdlib/gpu.hex"),
];
const SIN_EXAMPLES: &str = include_str!("../examples/sincos.hex");

/// Create a runtime with a provided user source file
fn runtime_with(source: &'static str) -> anyhow::Result<Runtime> {
    Runtime::from_sources(
        STDLIB.iter().copied().chain([source]),
        configured_gpu_dialect()?,
    )
    .map_err(Into::into)
}

fn configured_gpu_dialect() -> anyhow::Result<GpuDialect> {
    match std::env::var(GPU_DIALECT_ENV).as_deref() {
        Ok("hip") | Err(std::env::VarError::NotPresent) => Ok(GpuDialect::Hip),
        Ok("cuda") => Ok(GpuDialect::Cuda),
        Ok(value) => anyhow::bail!(
            "invalid GPU dialect `{value}` in {GPU_DIALECT_ENV}; expected `hip` or `cuda`"
        ),
        Err(std::env::VarError::NotUnicode(value)) => anyhow::bail!(
            "invalid GPU dialect in {GPU_DIALECT_ENV}: non-Unicode value {:?}",
            value
        ),
    }
}

#[test]
fn not_false() -> anyhow::Result<()> {
    let runtime = runtime_with(
        r#"
        (def program not : (bool val) -> (bool val) = bool.not)
        "#,
    )?;

    let [result] = runtime.exec("not", [false.into()])?;
    let Value::Bool(result) = result else {
        anyhow::bail!("not returned non-bool value: {result:?}");
    };

    assert_eq!(result, 1);
    Ok(())
}

#[test]
fn two_times_two() -> anyhow::Result<()> {
    let runtime = runtime_with(
        r#"
        (def program two-times-two : [] -> (u64 val) = (
          ({u64.one u64.one} u64.add)
          {[x . x x]}
          u64.mul
        ))
        "#,
    )?;

    let [result] = runtime.exec("two-times-two", [])?;
    let Value::U64(result) = result else {
        anyhow::bail!("two-times-two returned non-u64 value: {result:?}");
    };

    assert_eq!(result, 4);
    Ok(())
}

#[test]
fn two_times_two_u32() -> anyhow::Result<()> {
    let runtime = runtime_with(
        r#"
        (def program two-times-two : [] -> (u32 val) = (
          ({u32.one u32.one} u32.add)
          {[x . x x]}
          u32.mul
        ))
        "#,
    )?;

    let [result] = runtime.exec("two-times-two", [])?;
    let Value::U32(result) = result else {
        anyhow::bail!("two-times-two returned non-u32 value: {result:?}");
    };

    assert_eq!(result, 4);
    Ok(())
}

#[test]
fn two_times_two_float() -> anyhow::Result<()> {
    let runtime = runtime_with(
        r#"
        (def program two-times-two : [] -> (f32 val) = (
          ({f32.one f32.one} f32.add)
          {[x . x x]}
          f32.mul
        ))
        "#,
    )?;

    let [result] = runtime.exec("two-times-two", [])?;
    let Value::F32(result) = result else {
        anyhow::bail!("two-times-two returned non-f32 value: {result:?}");
    };

    assert_eq!(result, 4.0);
    Ok(())
}

#[test]
fn sin_approx_test() -> anyhow::Result<()> {
    let runtime = runtime_with(SIN_EXAMPLES)?;

    // Input range where the Taylor expansion is good enough
    for input in [0.0_f32, 0.5, 1.0, -0.5, -1.0] {
        let [result] = runtime.exec("sin-approx", [input.into()])?;
        let Value::F32(result) = result else {
            anyhow::bail!("sin-approx returned non-f32 value: {result:?}");
        };

        let expected = input.sin();
        assert!(
            (result - expected).abs() < 1e-4,
            "sin-approx({input}) = {result}, expected {expected}"
        );
    }

    Ok(())
}

#[test]
fn sin_approx_full_test() -> anyhow::Result<()> {
    let runtime = runtime_with(SIN_EXAMPLES)?;

    for input in [
        -200.0_f32, -100.0, -10.0, -6.0, -3.0, -1.9, -0.5, 0.0, 0.5, 3.0, 6.0, 10.0, 100.0, 200.0,
    ] {
        let [result] = runtime.exec("sin-approx-full", [input.into()])?;
        let Value::F32(result) = result else {
            anyhow::bail!("sin-approx-full returned non-f32 value: {result:?}");
        };

        let expected = input.sin();
        assert!(
            (result - expected).abs() < 1e-4,
            "sin-approx-full({input}) = {result}, expected {expected}"
        );
    }

    Ok(())
}

#[test]
fn u32_bitcast_f32_test() -> anyhow::Result<()> {
    let runtime = runtime_with(
        r#"
        (def program bitcast-one : [] -> (f32 val) = (
          const.u32.0x3F800000
          u32.bitcast-f32
        ))
        "#,
    )?;

    let [result] = runtime.exec("bitcast-one", [])?;
    let Value::F32(result) = result else {
        anyhow::bail!("bitcast-one returned non-f32 value: {result:?}");
    };

    assert_eq!(result, 1.0);
    Ok(())
}

#[test]
fn deadbeef_u64() -> anyhow::Result<()> {
    let runtime = runtime_with(
        r#"
        (def program deadbeef : [] -> (u64 val) = const.u64.0xDEADBEEFDEADBEEF)
        "#,
    )?;

    let [result] = runtime.exec("deadbeef", [])?;
    let Value::U64(result) = result else {
        anyhow::bail!("deadbeef returned non-u64 value: {result:?}");
    };

    assert_eq!(result, 0xDEADBEEFDEADBEEF_u64);
    Ok(())
}

#[test]
fn deadbeef_u32() -> anyhow::Result<()> {
    let runtime = runtime_with(
        r#"
        (def program deadbeef32 : [] -> (u32 val) = const.u32.0xDEADBEEF)
        "#,
    )?;

    let [result] = runtime.exec("deadbeef32", [])?;
    let Value::U32(result) = result else {
        anyhow::bail!("deadbeef32 returned non-u32 value: {result:?}");
    };

    assert_eq!(result, 0xDEADBEEF_u32);
    Ok(())
}

#[test]
fn array_head_u64() -> anyhow::Result<()> {
    let runtime = runtime_with(
        r#"
        (def program array-head-u64 : ([n.] (cap.own mem)) -> ([n.] (u64 val)) = (
          mem.cast.u64
          {
            (u64.assert-nz ix.zero)
            [b]
          }
          ix
        ))
        "#,
    )?;

    let values = [0x123456789abcdef0_u64, 7, 11];
    let input = runtime.mem_u64(&values)?;
    let [head] = runtime.exec("array-head-u64", [input])?;
    let Value::U64(head) = head else {
        anyhow::bail!("array-head-u64 returned non-u64 value: {head:?}");
    };

    assert_eq!(head, values[0]);
    Ok(())
}
