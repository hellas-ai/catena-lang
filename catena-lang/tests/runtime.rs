use catena_lang::runtime::{Runtime, Value};

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

/// Create a runtime with a provided user source file
fn runtime_with(source: &'static str) -> anyhow::Result<Runtime> {
    Runtime::from_sources(STDLIB.iter().copied().chain([source])).map_err(Into::into)
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
