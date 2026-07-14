use catena_lang::{
    codegen::GpuDialect,
    runtime::{Runtime, Value},
    stdlib,
};

const GPU_DIALECT_ENV: &str = "CATENA_GPU_DIALECT";

const SIN_EXAMPLES: &str = include_str!("../examples/sincos.hex");
const NN_EXAMPLES: &str = include_str!("../examples/nn.hex");
const SOFTMAX_EXAMPLES: &str = include_str!("../examples/softmax.hex");
const RMSNORM_EXAMPLES: &str = include_str!("../examples/rmsnorm.hex");
const REDUCEC_SUM_EXAMPLES: &str = include_str!("cases/reducec/sum.hex");

/// Create a runtime with a provided user source file
fn runtime_with(source: &'static str) -> anyhow::Result<Runtime> {
    Runtime::from_sources(stdlib::sources().chain([source]), configured_gpu_dialect()?)
        .map_err(Into::into)
}

fn runtime_with_sources(
    sources: impl IntoIterator<Item = &'static str>,
) -> anyhow::Result<Runtime> {
    Runtime::from_sources(stdlib::sources().chain(sources), configured_gpu_dialect()?)
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
fn u64_sub_basic() -> anyhow::Result<()> {
    let runtime = runtime_with(
        r#"
        (def program sub-basic : [] -> (u64 val) = (
          {u64.one u64.one}
          u64.add
          [two.]
          {[.two] u64.one}
          u64.sub
        ))
        "#,
    )?;

    let [result] = runtime.exec("sub-basic", [])?;
    let Value::U64(result) = result else {
        anyhow::bail!("sub-basic returned non-u64 value: {result:?}");
    };

    assert_eq!(result, 1);
    Ok(())
}

#[test]
fn u64_cmp_ops_test() -> anyhow::Result<()> {
    let runtime = runtime_with(
        r#"
        (def program u64-ne-test : [] -> (bool val) = (
          {[.]
            (const.u64.0x0000000000000002 [lhs.])
            (const.u64.0x0000000000000003 [rhs.])
            ([.lhs rhs] u64.ne [result.])
            [.result]
          }
        ))
        (def program u64-lt-test : [] -> (bool val) = (
          {[.]
            (const.u64.0x0000000000000002 [lhs.])
            (const.u64.0x0000000000000003 [rhs.])
            ([.lhs rhs] u64.lt [result.])
            [.result]
          }
        ))
        (def program u64-lte-test : [] -> (bool val) = (
          {[.]
            (const.u64.0x0000000000000002 [lhs.])
            (const.u64.0x0000000000000003 [rhs.])
            ([.lhs rhs] u64.lte [result.])
            [.result]
          }
        ))
        (def program u64-gte-test : [] -> (bool val) = (
          {[.]
            (const.u64.0x0000000000000002 [lhs.])
            (const.u64.0x0000000000000003 [rhs.])
            ([.lhs rhs] u64.gte [result.])
            [.result]
          }
        ))
        "#,
    )?;

    for (name, expected) in [
        ("u64-ne-test", 1_u8),
        ("u64-lt-test", 1_u8),
        ("u64-lte-test", 1_u8),
        ("u64-gte-test", 0_u8),
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
fn f32_fma_basic_test() -> anyhow::Result<()> {
    let runtime = runtime_with(
        r#"
        (def program fma-basic : [] -> (f32 val) = (
          {f32.one f32.one f32.one}
          f32.fma
        ))
        "#,
    )?;

    let [result] = runtime.exec("fma-basic", [])?;
    let Value::F32(result) = result else {
        anyhow::bail!("fma-basic returned non-f32 value: {result:?}");
    };

    assert_eq!(result, 2.0);
    Ok(())
}

#[test]
fn f32_fma_is_fused_test() -> anyhow::Result<()> {
    let a = f32::from_bits(0x3F800001);
    let b = f32::from_bits(0x3F800001);
    let c = f32::from_bits(0x33800000);
    let fused_bits = a.mul_add(b, c).to_bits();
    let separate_bits = ((a * b) + c).to_bits();

    assert_eq!(fused_bits, 0x3F800003);
    assert_eq!(separate_bits, 0x3F800002);

    let runtime = runtime_with(
        r#"
        (def program fma-fused-bits : {(f32 val) (f32 val) (f32 val)} -> (u32 val) = (
          {[a b c.]
            ([.a b c] f32.fma [result.])
            ([.result] f32.bitcast-u32 [bits.])
            [.bits]
          }
        ))
        "#,
    )?;

    let [result] = runtime.exec("fma-fused-bits", [a.into(), b.into(), c.into()])?;
    let Value::U32(result) = result else {
        anyhow::bail!("fma-fused-bits returned non-u32 value: {result:?}");
    };

    assert_eq!(result, fused_bits);
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
    let runtime = runtime_with(NN_EXAMPLES)?;

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
fn exp2_approx_test() -> anyhow::Result<()> {
    let runtime = runtime_with(NN_EXAMPLES)?;

    for input in [-3.0_f32, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0] {
        let [result] = runtime.exec("exp2-approx", [input.into()])?;
        let Value::F32(result) = result else {
            anyhow::bail!("exp2-approx returned non-f32 value: {result:?}");
        };

        let expected = input.exp2();
        let error = (result - expected).abs() / expected.max(1.0);
        assert!(
            error < 4e-3,
            "exp2-approx({input}) = {result}, expected {expected}, rel-ish error {error}"
        );
    }

    Ok(())
}

#[path = "cases/materializec.rs"]
mod materializec;

#[test]
fn sigmoid_test() -> anyhow::Result<()> {
    let runtime = runtime_with(NN_EXAMPLES)?;

    for input in [-6.0_f32, -1.0, 0.0, 1.0, 6.0] {
        let [result] = runtime.exec("sigmoid", [input.into()])?;
        let Value::F32(result) = result else {
            anyhow::bail!("sigmoid returned non-f32 value: {result:?}");
        };

        let expected = 1.0 / (1.0 + (-input).exp());
        let error = (result - expected).abs();
        assert!(
            error < 4e-3,
            "sigmoid({input}) = {result}, expected {expected}, abs error {error}"
        );
    }

    Ok(())
}

#[test]
fn silu_test() -> anyhow::Result<()> {
    let runtime = runtime_with(NN_EXAMPLES)?;

    for input in [-3.0_f32, -1.0, 0.0, 1.0, 3.0] {
        let [result] = runtime.exec("silu", [input.into()])?;
        let Value::F32(result) = result else {
            anyhow::bail!("silu returned non-f32 value: {result:?}");
        };

        let sigmoid = 1.0 / (1.0 + (-input).exp());
        let expected = input * sigmoid;
        let error = (result - expected).abs();
        assert!(
            error < 2e-2,
            "silu({input}) = {result}, expected {expected}, abs error {error}"
        );
    }

    Ok(())
}

#[test]
fn tanh_test() -> anyhow::Result<()> {
    let runtime = runtime_with(NN_EXAMPLES)?;

    for input in [-3.0_f32, -1.0, 0.0, 1.0, 3.0] {
        let [result] = runtime.exec("tanh", [input.into()])?;
        let Value::F32(result) = result else {
            anyhow::bail!("tanh returned non-f32 value: {result:?}");
        };

        let expected = input.tanh();
        let error = (result - expected).abs();
        assert!(
            error < 8e-3,
            "tanh({input}) = {result}, expected {expected}, abs error {error}"
        );
    }

    Ok(())
}

#[test]
fn gelu_approx_test() -> anyhow::Result<()> {
    let runtime = runtime_with(NN_EXAMPLES)?;

    for input in [-3.0_f32, -1.0, 0.0, 1.0, 3.0] {
        let [result] = runtime.exec("gelu-approx", [input.into()])?;
        let Value::F32(result) = result else {
            anyhow::bail!("gelu-approx returned non-f32 value: {result:?}");
        };

        let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
        let expected =
            0.5 * input * (1.0 + (sqrt_2_over_pi * (input + 0.044_715 * input.powi(3))).tanh());
        let error = (result - expected).abs();
        assert!(
            error < 2e-2,
            "gelu-approx({input}) = {result}, expected {expected}, abs error {error}"
        );
    }

    Ok(())
}

#[test]
fn sqrt_test() -> anyhow::Result<()> {
    let runtime = runtime_with(NN_EXAMPLES)?;

    for input in [0.0_f32, 0.25, 1.0, 2.0, 9.0, 100.0] {
        let [result] = runtime.exec("sqrt", [input.into()])?;
        let Value::F32(result) = result else {
            anyhow::bail!("sqrt returned non-f32 value: {result:?}");
        };

        let expected = input.sqrt();
        let error = (result - expected).abs();
        assert!(
            error < 1e-4,
            "sqrt({input}) = {result}, expected {expected}, abs error {error}"
        );
    }

    Ok(())
}

#[test]
fn log_approx_test() -> anyhow::Result<()> {
    let runtime = runtime_with(NN_EXAMPLES)?;

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

#[test]
fn log2_approx_test() -> anyhow::Result<()> {
    let runtime = runtime_with(NN_EXAMPLES)?;

    for input in [0.1_f32, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 8.0, 10.0] {
        let [result] = runtime.exec("log2-approx", [input.into()])?;
        let Value::F32(result) = result else {
            anyhow::bail!("log2-approx returned non-f32 value: {result:?}");
        };

        let expected = input.log2();
        let error = (result - expected).abs();
        assert!(
            error < 1e-3,
            "log2-approx({input}) = {result}, expected {expected}, abs error {error}"
        );
    }

    Ok(())
}

#[test]
fn powf_test() -> anyhow::Result<()> {
    let runtime = runtime_with(NN_EXAMPLES)?;

    for (base, exponent) in [
        (0.25_f32, 0.5_f32),
        (0.5_f32, 2.0_f32),
        (1.0_f32, 3.0_f32),
        (1.5_f32, -1.0_f32),
        (2.0_f32, 3.0_f32),
        (3.0_f32, 0.5_f32),
        (10.0_f32, 0.25_f32),
    ] {
        let [result] = runtime.exec("powf", [base.into(), exponent.into()])?;
        let Value::F32(result) = result else {
            anyhow::bail!("powf returned non-f32 value: {result:?}");
        };

        let expected = base.powf(exponent);
        let error = (result - expected).abs() / expected.abs().max(1.0);
        assert!(
            error < 8e-3,
            "powf({base}, {exponent}) = {result}, expected {expected}, rel-ish error {error}"
        );
    }

    Ok(())
}

#[test]
fn softmax_test() -> anyhow::Result<()> {
    let runtime = runtime_with_sources([REDUCEC_SUM_EXAMPLES, NN_EXAMPLES, SOFTMAX_EXAMPLES])?;

    let input_values = [1.0_f32, 2.0, 4.0];
    let input = runtime.mem_f32(&input_values)?;
    let [result] = runtime.exec("softmax", [input])?;
    let Value::Mem(result) = result else {
        anyhow::bail!("softmax returned non-mem value: {result:?}");
    };

    let values = result.to_f32_vec();
    assert_eq!(values.len(), input_values.len());

    let max = input_values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let denom: f32 = input_values.iter().map(|x| (x - max).exp()).sum();
    let expected: Vec<f32> = input_values
        .iter()
        .map(|x| (x - max).exp() / denom)
        .collect();

    for (actual, expected) in values.iter().zip(expected.iter()) {
        let error = (actual - expected).abs();
        assert!(
            error < 6e-3,
            "softmax output {actual} differed from expected {expected} by {error}"
        );
    }

    let sum: f32 = values.iter().sum();
    assert!(
        (sum - 1.0).abs() < 6e-3,
        "softmax outputs should sum to 1, got {sum}"
    );
    Ok(())
}

#[test]
fn rmsnorm_test() -> anyhow::Result<()> {
    let runtime = runtime_with_sources([REDUCEC_SUM_EXAMPLES, NN_EXAMPLES, RMSNORM_EXAMPLES])?;

    let input_values = [1.0_f32, 2.0, 4.0, -1.0, -9.0, 13.29];
    let input = runtime.mem_f32(&input_values)?;
    let [result] = runtime.exec("rmsnorm", [input])?;
    let Value::Mem(result) = result else {
        anyhow::bail!("rmsnorm returned non-mem value: {result:?}");
    };

    let values = result.to_f32_vec();
    assert_eq!(values.len(), input_values.len());

    let mean_sq: f32 = input_values.iter().map(|x| x * x).sum::<f32>() / input_values.len() as f32;
    let rms = (mean_sq + 1e-5_f32).sqrt();
    let expected: Vec<f32> = input_values.iter().map(|x| x / rms).collect();

    for (actual, expected) in values.iter().zip(expected.iter()) {
        let error = (actual - expected).abs();
        assert!(
            error < 3e-3,
            "rmsnorm output {actual} differed from expected {expected} by {error}"
        );
    }

    let output_mean_sq: f32 = values.iter().map(|x| x * x).sum::<f32>() / values.len() as f32;
    let output_rms = output_mean_sq.sqrt();
    assert!(
        (output_rms - 1.0).abs() < 3e-3,
        "rmsnorm outputs should have unit RMS, got {output_rms}"
    );
    Ok(())
}

#[test]
fn broadcast_f32_test() -> anyhow::Result<()> {
    let runtime = runtime_with(NN_EXAMPLES)?;

    let [empty] = runtime.exec("broadcast-f32", [3.5_f32.into(), 0_u64.into()])?;
    let Value::Mem(empty) = empty else {
        anyhow::bail!("broadcast-f32 returned non-mem value: {empty:?}");
    };
    assert_eq!(empty.to_f32_vec(), Vec::<f32>::new());

    let [result] = runtime.exec("broadcast-f32", [3.5_f32.into(), 4_u64.into()])?;
    let Value::Mem(result) = result else {
        anyhow::bail!("broadcast-f32 returned non-mem value: {result:?}");
    };
    assert_eq!(result.to_f32_vec(), vec![3.5, 3.5, 3.5, 3.5]);
    Ok(())
}

#[test]
fn arange_f32_test() -> anyhow::Result<()> {
    let runtime = runtime_with(NN_EXAMPLES)?;

    let [empty] = runtime.exec("arange-f32", [0_u64.into()])?;
    let Value::Mem(empty) = empty else {
        anyhow::bail!("arange-f32 returned non-mem value: {empty:?}");
    };
    assert_eq!(empty.to_f32_vec(), Vec::<f32>::new());

    let [result] = runtime.exec("arange-f32", [5_u64.into()])?;
    let Value::Mem(result) = result else {
        anyhow::bail!("arange-f32 returned non-mem value: {result:?}");
    };
    assert_eq!(result.to_f32_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    Ok(())
}

#[test]
fn slice_f32_test() -> anyhow::Result<()> {
    let runtime = runtime_with(NN_EXAMPLES)?;

    let [result] = runtime.exec(
        "slice-f32",
        [
            runtime.mem_f32(&[10.0_f32, 20.0, 30.0, 40.0])?,
            0_u64.into(),
            2_u64.into(),
        ],
    )?;
    let Value::Mem(result) = result else {
        anyhow::bail!("slice-f32 returned non-mem value: {result:?}");
    };
    assert_eq!(result.to_f32_vec(), vec![10.0, 20.0]);

    let [result] = runtime.exec(
        "slice-f32",
        [
            runtime.mem_f32(&[10.0_f32, 20.0, 30.0, 40.0])?,
            1_u64.into(),
            2_u64.into(),
        ],
    )?;
    let Value::Mem(result) = result else {
        anyhow::bail!("slice-f32 returned non-mem value: {result:?}");
    };
    assert_eq!(result.to_f32_vec(), vec![20.0, 30.0]);

    let [result] = runtime.exec(
        "slice-f32",
        [
            runtime.mem_f32(&[10.0_f32, 20.0, 30.0, 40.0])?,
            2_u64.into(),
            0_u64.into(),
        ],
    )?;
    let Value::Mem(result) = result else {
        anyhow::bail!("slice-f32 returned non-mem value: {result:?}");
    };
    assert_eq!(result.to_f32_vec(), Vec::<f32>::new());

    let [result] = runtime.exec(
        "slice-f32",
        [
            runtime.mem_f32(&[10.0_f32, 20.0, 30.0, 40.0])?,
            0_u64.into(),
            4_u64.into(),
        ],
    )?;
    let Value::Mem(result) = result else {
        anyhow::bail!("slice-f32 returned non-mem value: {result:?}");
    };
    assert_eq!(result.to_f32_vec(), vec![10.0, 20.0, 30.0, 40.0]);
    Ok(())
}

#[test]
fn argmax_f32_test() -> anyhow::Result<()> {
    let runtime = runtime_with(NN_EXAMPLES)?;

    for (input_values, expected) in [
        (vec![5.0_f32], 0_u64),
        (vec![1.0_f32, 2.0, 4.0], 2_u64),
        (vec![4.0_f32, 2.0, 1.0], 0_u64),
        (vec![1.0_f32, 7.0, 3.0, 2.0], 1_u64),
        (vec![1.0_f32, 7.0, 7.0, 2.0], 1_u64),
        (vec![3.0_f32, 3.0, 3.0], 0_u64),
    ] {
        let input = runtime.mem_f32(&input_values)?;
        let [result] = runtime.exec("argmax-f32", [input])?;
        let Value::U64(result) = result else {
            anyhow::bail!("argmax-f32 returned non-u64 value: {result:?}");
        };
        assert_eq!(result, expected, "argmax-f32({input_values:?})");
    }

    Ok(())
}

#[path = "cases/reducec.rs"]
mod reducec;

#[path = "cases/reduce.rs"]
mod reduce;

#[path = "cases/closure.rs"]
mod closure;
