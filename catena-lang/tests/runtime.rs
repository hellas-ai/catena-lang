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
const LOG_EXAMPLES: &str = include_str!("../examples/log.hex");
const SOFTMAX_EXAMPLES: &str = include_str!("../examples/softmax.hex");

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
fn f32_bitcast_u32_test() -> anyhow::Result<()> {
    let runtime = runtime_with(
        r#"
        (def program bitcast-one-bits : [] -> (u32 val) = (
          f32.one
          f32.bitcast-u32
        ))
        "#,
    )?;

    let [result] = runtime.exec("bitcast-one-bits", [])?;
    let Value::U32(result) = result else {
        anyhow::bail!("bitcast-one-bits returned non-u32 value: {result:?}");
    };

    assert_eq!(result, 0x3F800000);
    Ok(())
}

#[test]
fn u32_shift_and_sub_test() -> anyhow::Result<()> {
    let runtime = runtime_with(
        r#"
        (def program shift-and-sub : [] -> (u32 val) = (
          {[.]
            (const.u32.0x00000020 [lhs.])
            (const.u32.0x00000003 [shift.])
            (const.u32.0x00000001 [one.])
            ([.lhs shift] u32.shr [shr.])
            ([.one shift] u32.shl [shl.])
            ([.shr shl] u32.sub [result.])
            [.result]
          }
        ))
        "#,
    )?;

    let [result] = runtime.exec("shift-and-sub", [])?;
    let Value::U32(result) = result else {
        anyhow::bail!("shift-and-sub returned non-u32 value: {result:?}");
    };

    assert_eq!(result, 0xFFFF_FFFC);
    Ok(())
}

#[test]
fn u32_bitwise_ops_test() -> anyhow::Result<()> {
    let runtime = runtime_with(
        r#"
        (def program u32-and-test : [] -> (u32 val) = (
          {[.]
            (const.u32.0x00FF00FF [lhs.])
            (const.u32.0x0F0F0F0F [rhs.])
            ([.lhs rhs] u32.and [result.])
            [.result]
          }
        ))
        (def program u32-or-test : [] -> (u32 val) = (
          {[.]
            (const.u32.0x00FF00FF [lhs.])
            (const.u32.0x0F0F0F0F [rhs.])
            ([.lhs rhs] u32.or [result.])
            [.result]
          }
        ))
        (def program u32-xor-test : [] -> (u32 val) = (
          {[.]
            (const.u32.0x00FF00FF [lhs.])
            (const.u32.0x0F0F0F0F [rhs.])
            ([.lhs rhs] u32.xor [result.])
            [.result]
          }
        ))
        (def program u32-not-test : [] -> (u32 val) = (
          {[.]
            (const.u32.0x00FF00FF [value.])
            ([.value] u32.not [result.])
            [.result]
          }
        ))
        "#,
    )?;

    for (name, expected) in [
        ("u32-and-test", 0x000F000F_u32),
        ("u32-or-test", 0x0FFF0FFF_u32),
        ("u32-xor-test", 0x0FF00FF0_u32),
        ("u32-not-test", 0xFF00FF00_u32),
    ] {
        let [result] = runtime.exec(name, [])?;
        let Value::U32(result) = result else {
            anyhow::bail!("{name} returned non-u32 value: {result:?}");
        };
        assert_eq!(
            result, expected,
            "{name} returned {result:#x}, expected {expected:#x}"
        );
    }

    Ok(())
}

#[test]
fn u32_cmp_ops_test() -> anyhow::Result<()> {
    let runtime = runtime_with(
        r#"
        (def program u32-eq-test : [] -> (bool val) = (
          {[.]
            (const.u32.0x00000002 [lhs.])
            (const.u32.0x00000003 [rhs.])
            ([.lhs rhs] u32.eq [result.])
            [.result]
          }
        ))
        (def program u32-ne-test : [] -> (bool val) = (
          {[.]
            (const.u32.0x00000002 [lhs.])
            (const.u32.0x00000003 [rhs.])
            ([.lhs rhs] u32.ne [result.])
            [.result]
          }
        ))
        (def program u32-lt-test : [] -> (bool val) = (
          {[.]
            (const.u32.0x00000002 [lhs.])
            (const.u32.0x00000003 [rhs.])
            ([.lhs rhs] u32.lt [result.])
            [.result]
          }
        ))
        (def program u32-gt-test : [] -> (bool val) = (
          {[.]
            (const.u32.0x00000002 [lhs.])
            (const.u32.0x00000003 [rhs.])
            ([.lhs rhs] u32.gt [result.])
            [.result]
          }
        ))
        (def program u32-lte-test : [] -> (bool val) = (
          {[.]
            (const.u32.0x00000002 [lhs.])
            (const.u32.0x00000003 [rhs.])
            ([.lhs rhs] u32.lte [result.])
            [.result]
          }
        ))
        (def program u32-gte-test : [] -> (bool val) = (
          {[.]
            (const.u32.0x00000002 [lhs.])
            (const.u32.0x00000003 [rhs.])
            ([.lhs rhs] u32.gte [result.])
            [.result]
          }
        ))
        "#,
    )?;

    for (name, expected) in [
        ("u32-eq-test", 0_u8),
        ("u32-ne-test", 1_u8),
        ("u32-lt-test", 1_u8),
        ("u32-gt-test", 0_u8),
        ("u32-lte-test", 1_u8),
        ("u32-gte-test", 0_u8),
    ] {
        let [result] = runtime.exec(name, [])?;
        let Value::Bool(result) = result else {
            anyhow::bail!("{name} returned non-bool value: {result:?}");
        };
        assert_eq!(
            result, expected,
            "{name} returned {result}, expected {expected}"
        );
    }
    Ok(())
}

#[test]
fn f32_cmp_ops_test() -> anyhow::Result<()> {
    let runtime = runtime_with(
        r#"
        (def program f32-lt-test : [] -> (bool val) = (
          {[.]
            (const.u32.0x3FC00000 u32.bitcast-f32 [lhs.])
            (const.u32.0x40200000 u32.bitcast-f32 [rhs.])
            ([.lhs rhs] f32.lt [result.])
            [.result]
          }
        ))
        (def program f32-eq-test : [] -> (bool val) = (
          {[.]
            (const.u32.0x3FC00000 u32.bitcast-f32 [lhs.])
            (const.u32.0x40200000 u32.bitcast-f32 [rhs.])
            ([.lhs rhs] f32.eq [result.])
            [.result]
          }
        ))
        (def program f32-ne-test : [] -> (bool val) = (
          {[.]
            (const.u32.0x3FC00000 u32.bitcast-f32 [lhs.])
            (const.u32.0x40200000 u32.bitcast-f32 [rhs.])
            ([.lhs rhs] f32.ne [result.])
            [.result]
          }
        ))
        (def program f32-gt-test : [] -> (bool val) = (
          {[.]
            (const.u32.0x3FC00000 u32.bitcast-f32 [lhs.])
            (const.u32.0x40200000 u32.bitcast-f32 [rhs.])
            ([.lhs rhs] f32.gt [result.])
            [.result]
          }
        ))
        (def program f32-lte-test : [] -> (bool val) = (
          {[.]
            (const.u32.0x3FC00000 u32.bitcast-f32 [lhs.])
            (const.u32.0x40200000 u32.bitcast-f32 [rhs.])
            ([.lhs rhs] f32.lte [result.])
            [.result]
          }
        ))
        (def program f32-gte-test : [] -> (bool val) = (
          {[.]
            (const.u32.0x3FC00000 u32.bitcast-f32 [lhs.])
            (const.u32.0x40200000 u32.bitcast-f32 [rhs.])
            ([.lhs rhs] f32.gte [result.])
            [.result]
          }
        ))
        "#,
    )?;

    for (name, expected) in [
        ("f32-lt-test", 1_u8),
        ("f32-eq-test", 0_u8),
        ("f32-ne-test", 1_u8),
        ("f32-gt-test", 0_u8),
        ("f32-lte-test", 1_u8),
        ("f32-gte-test", 0_u8),
    ] {
        let [result] = runtime.exec(name, [])?;
        let Value::Bool(result) = result else {
            anyhow::bail!("{name} returned non-bool value: {result:?}");
        };
        assert_eq!(
            result, expected,
            "{name} returned {result}, expected {expected}"
        );
    }
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

#[test]
fn exp_approx_test() -> anyhow::Result<()> {
    let runtime = runtime_with(SOFTMAX_EXAMPLES)?;

    for input in [-3.0_f32, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0] {
        let [result] = runtime.exec("exp-approx", [input.into()])?;
        let Value::F32(result) = result else {
            anyhow::bail!("exp-approx returned non-f32 value: {result:?}");
        };

        let expected = input.exp();
        let error = (result - expected).abs() / expected.max(1.0);
        assert!(
            error < 4e-3,
            "exp-approx({input}) = {result}, expected {expected}, rel-ish error {error}"
        );
    }

    Ok(())
}

#[test]
fn log_approx_test() -> anyhow::Result<()> {
    let runtime = runtime_with(LOG_EXAMPLES)?;

    for input in [0.1_f32, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 8.0, 10.0] {
        let [result] = runtime.exec("log-approx", [input.into()])?;
        let Value::F32(result) = result else {
            anyhow::bail!("log-approx returned non-f32 value: {result:?}");
        };

        let expected = input.ln();
        let error = (result - expected).abs();
        assert!(
            error < 6e-4,
            "log-approx({input}) = {result}, expected {expected}, abs error {error}"
        );
    }

    Ok(())
}
