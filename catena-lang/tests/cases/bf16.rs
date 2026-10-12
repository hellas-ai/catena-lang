use super::*;

const MATMUL_SOURCE: &str = include_str!("bf16/matmul.hex");

fn bf16_values(values: &[f32]) -> Vec<half::bf16> {
    values.iter().copied().map(half::bf16::from_f32).collect()
}

fn bf16_bits(values: &[half::bf16]) -> Vec<u16> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn bf16_values_from_bits(bits: &[u16]) -> Vec<half::bf16> {
    bits.iter().copied().map(half::bf16::from_bits).collect()
}

#[test]
fn bf16_matmul_row_major_bufs_from_mems() -> anyhow::Result<()> {
    let runtime = runtime_with(MATMUL_SOURCE)?;
    let a_values = bf16_values(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b_values = bf16_values(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let a = runtime.mem_u16(&bf16_bits(&a_values))?;
    let b = runtime.mem_u16(&bf16_bits(&b_values))?;

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

    assert_eq!(
        bf16_values_from_bits(&result.to_u16_vec()),
        bf16_values(&[58.0, 64.0, 139.0, 154.0])
    );
    Ok(())
}

#[test]
fn bf16_matmul_right_identity_view() -> anyhow::Result<()> {
    let runtime = runtime_with(MATMUL_SOURCE)?;
    let input_values = bf16_values(&[1.0, 2.0, 3.0, 4.0]);
    let input = runtime.mem_u16(&bf16_bits(&input_values))?;
    let [result] = runtime.exec(
        "matmul-right-identity-2x2-via-mem",
        [input.as_ref().into(), 2_u64.into(), 4_u64.into()],
    )?;
    let Value::MemOwn(result) = result else {
        anyhow::bail!("matmul-right-identity-2x2-via-mem returned non-mem value: {result:?}");
    };

    assert_eq!(bf16_values_from_bits(&result.to_u16_vec()), input_values);
    Ok(())
}

#[test]
fn bf16_matmul_left_identity_view() -> anyhow::Result<()> {
    let runtime = runtime_with(MATMUL_SOURCE)?;
    let input_values = bf16_values(&[1.0, 2.0, 3.0, 4.0]);
    let input = runtime.mem_u16(&bf16_bits(&input_values))?;
    let [result] = runtime.exec(
        "matmul-left-identity-2x2-via-mem",
        [input.as_ref().into(), 2_u64.into(), 4_u64.into()],
    )?;
    let Value::MemOwn(result) = result else {
        anyhow::bail!("matmul-left-identity-2x2-via-mem returned non-mem value: {result:?}");
    };

    assert_eq!(bf16_values_from_bits(&result.to_u16_vec()), input_values);
    Ok(())
}

#[test]
fn bf16_core_numeric_ops() -> anyhow::Result<()> {
    use half::bf16;

    // U16-adapted forms of the BF16 operations, suitable for runtime entrypoints.
    // The bitcasts keep BF16 itself inside generated Catena code and off the ABI.
    let runtime = runtime_with(
        r#"
        (def program bf16-one : [] -> (u16 val) = (bf16.one bf16.bitcast-u16))
        (def program bf16-add : {(u16 val) (u16 val)} -> (u16 val) =
          ({u16.bitcast-bf16 u16.bitcast-bf16} bf16.add bf16.bitcast-u16))
        (def program bf16-sub : {(u16 val) (u16 val)} -> (u16 val) =
          ({u16.bitcast-bf16 u16.bitcast-bf16} bf16.sub bf16.bitcast-u16))
        (def program bf16-neg : (u16 val) -> (u16 val) =
          (u16.bitcast-bf16 bf16.neg bf16.bitcast-u16))
        (def program bf16-mul : {(u16 val) (u16 val)} -> (u16 val) =
          ({u16.bitcast-bf16 u16.bitcast-bf16} bf16.mul bf16.bitcast-u16))
        (def program bf16-fma : {(u16 val) (u16 val) (u16 val)} -> (u16 val) =
          ({u16.bitcast-bf16 u16.bitcast-bf16 u16.bitcast-bf16}
           bf16.fma bf16.bitcast-u16))
        (def program bf16-div : {(u16 val) (u16 val)} -> (u16 val) =
          ({u16.bitcast-bf16 u16.bitcast-bf16} bf16.div bf16.bitcast-u16))
        (def program bf16-select : {(bool val) (u16 val) (u16 val)} -> (u16 val) =
          ({bool.id u16.bitcast-bf16 u16.bitcast-bf16}
           bf16.select bf16.bitcast-u16))
        (def program bf16-to-f32 : (u16 val) -> (f32 val) =
          (u16.bitcast-bf16 bf16.to-f32))
        (def program f32-to-bf16 : (f32 val) -> (u16 val) =
          (f32.to-bf16 bf16.bitcast-u16))
        "#,
    )?;

    let a = bf16::from_f32(1.5);
    let b = bf16::from_f32(0.75);
    let c = bf16::from_f32(-0.5);

    let [Value::U16(one)] = runtime.exec("bf16-one", [])? else {
        anyhow::bail!("bf16-one returned the wrong value kind");
    };
    assert_eq!(one, bf16::ONE.to_bits());

    for (name, expected) in [
        ("bf16-add", bf16::from_f32(a.to_f32() + b.to_f32())),
        ("bf16-sub", bf16::from_f32(a.to_f32() - b.to_f32())),
        ("bf16-mul", bf16::from_f32(a.to_f32() * b.to_f32())),
        ("bf16-div", bf16::from_f32(a.to_f32() / b.to_f32())),
    ] {
        let [Value::U16(actual)] = runtime.exec(name, [a.to_bits().into(), b.to_bits().into()])?
        else {
            anyhow::bail!("{name} returned the wrong value kind");
        };
        assert_eq!(actual, expected.to_bits(), "{name}");
    }

    let [Value::U16(negated)] = runtime.exec("bf16-neg", [a.to_bits().into()])? else {
        anyhow::bail!("bf16-neg returned the wrong value kind");
    };
    assert_eq!(negated, bf16::from_f32(-a.to_f32()).to_bits());

    let [Value::U16(fused)] = runtime.exec(
        "bf16-fma",
        [a.to_bits().into(), b.to_bits().into(), c.to_bits().into()],
    )?
    else {
        anyhow::bail!("bf16-fma returned the wrong value kind");
    };
    assert_eq!(
        fused,
        bf16::from_f32(a.to_f32().mul_add(b.to_f32(), c.to_f32())).to_bits()
    );

    let [Value::U16(selected)] = runtime.exec(
        "bf16-select",
        [true.into(), a.to_bits().into(), b.to_bits().into()],
    )?
    else {
        anyhow::bail!("bf16-select returned the wrong value kind");
    };
    assert_eq!(selected, a.to_bits());

    let [Value::F32(as_f32)] = runtime.exec("bf16-to-f32", [a.to_bits().into()])? else {
        anyhow::bail!("bf16-to-f32 returned the wrong value kind");
    };
    assert_eq!(as_f32.to_bits(), a.to_f32().to_bits());

    let midpoint = f32::from_bits(0x3f80_8000);
    let [Value::U16(rounded)] = runtime.exec("f32-to-bf16", [midpoint.into()])? else {
        anyhow::bail!("f32-to-bf16 returned the wrong value kind");
    };
    assert_eq!(rounded, bf16::from_f32(midpoint).to_bits());

    for special in [f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0, f32::NAN] {
        let [Value::U16(actual)] = runtime.exec("f32-to-bf16", [special.into()])? else {
            anyhow::bail!("f32-to-bf16 returned the wrong value kind");
        };
        if special.is_nan() {
            assert!(bf16::from_bits(actual).is_nan());
        } else {
            assert_eq!(actual, bf16::from_f32(special).to_bits());
        }
    }

    Ok(())
}

#[test]
fn bf16_memory_round_trip_and_cast() -> anyhow::Result<()> {
    use half::bf16;

    let runtime = runtime_with(
        r#"
        (def program array-head-bf16 : ([n.] (cap.ref mem)) -> ([n.] (u16 val)) = (
          mem.cast.bf16
          {
            (u64.assert-nz ix.zero)
            [b]
          }
          ix
          bf16.bitcast-u16
        ))
        "#,
    )?;
    let values = [bf16::from_f32(1.5), bf16::from_f32(-2.25)];
    let bits = bf16_bits(&values);
    let memory = runtime.mem_u16(&bits)?;
    assert_eq!(memory.try_to_u16_vec()?, bits);

    let [Value::U16(head)] = runtime.exec("array-head-bf16", [memory.as_ref().into()])? else {
        anyhow::bail!("array-head-bf16 returned the wrong value kind");
    };
    assert_eq!(head, values[0].to_bits());
    Ok(())
}

#[test]
fn bf16_fma_is_fused_test() -> anyhow::Result<()> {
    use half::bf16;

    let a = bf16::from_bits(0x3fc4);
    let b = bf16::from_bits(0x4005);
    let c = bf16::from_bits(0x3f1e);
    let fused = bf16::from_f32(a.to_f32().mul_add(b.to_f32(), c.to_f32()));
    let product = bf16::from_f32(a.to_f32() * b.to_f32());
    let separate = bf16::from_f32(product.to_f32() + c.to_f32());
    assert_eq!(fused.to_bits(), 0x4073);
    assert_eq!(separate.to_bits(), 0x4074);

    let runtime = runtime_with(
        r#"
        (def program bf16-fma-fused :
          {(u16 val) (u16 val) (u16 val)} -> (u16 val)
        = ({u16.bitcast-bf16 u16.bitcast-bf16 u16.bitcast-bf16}
           bf16.fma bf16.bitcast-u16))
        "#,
    )?;
    let [Value::U16(result)] = runtime.exec(
        "bf16-fma-fused",
        [a.to_bits().into(), b.to_bits().into(), c.to_bits().into()],
    )?
    else {
        anyhow::bail!("bf16-fma-fused returned the wrong value kind");
    };
    assert_eq!(result, fused.to_bits());
    Ok(())
}

#[test]
fn bf16_cmp_ops_test() -> anyhow::Result<()> {
    let runtime = runtime_with(
        r#"
        (def program bf16-lt-test : [] -> (bool val) = (
          {(const.u32.0x3FC00000 u32.bitcast-f32 f32.to-bf16)
           (const.u32.0x40200000 u32.bitcast-f32 f32.to-bf16)}
          bf16.lt))
        (def program bf16-eq-test : [] -> (bool val) = (
          {(const.u32.0x3FC00000 u32.bitcast-f32 f32.to-bf16)
           (const.u32.0x40200000 u32.bitcast-f32 f32.to-bf16)}
          bf16.eq))
        (def program bf16-ne-test : [] -> (bool val) = (
          {(const.u32.0x3FC00000 u32.bitcast-f32 f32.to-bf16)
           (const.u32.0x40200000 u32.bitcast-f32 f32.to-bf16)}
          bf16.ne))
        (def program bf16-gt-test : [] -> (bool val) = (
          {(const.u32.0x3FC00000 u32.bitcast-f32 f32.to-bf16)
           (const.u32.0x40200000 u32.bitcast-f32 f32.to-bf16)}
          bf16.gt))
        (def program bf16-lte-test : [] -> (bool val) = (
          {(const.u32.0x3FC00000 u32.bitcast-f32 f32.to-bf16)
           (const.u32.0x40200000 u32.bitcast-f32 f32.to-bf16)}
          bf16.lte))
        (def program bf16-gte-test : [] -> (bool val) = (
          {(const.u32.0x3FC00000 u32.bitcast-f32 f32.to-bf16)
           (const.u32.0x40200000 u32.bitcast-f32 f32.to-bf16)}
          bf16.gte))
        "#,
    )?;

    for (name, expected) in [
        ("bf16-lt-test", 1_u8),
        ("bf16-eq-test", 0_u8),
        ("bf16-ne-test", 1_u8),
        ("bf16-gt-test", 0_u8),
        ("bf16-lte-test", 1_u8),
        ("bf16-gte-test", 0_u8),
    ] {
        let [Value::Bool(result)] = runtime.exec(name, [])? else {
            anyhow::bail!("{name} returned the wrong value kind");
        };
        assert_eq!(result, expected, "{name}");
    }
    Ok(())
}

#[test]
fn broadcast_bf16_test() -> anyhow::Result<()> {
    let runtime = runtime_with(
        r#"
        (def program tensor.broadcast-bf16-from-u16 :
          {(u16 val) ({[n] u64} :)} -> ([n.] (cap.own mem))
        = ([bits len.]
          ([.bits] u16.bitcast-bf16 [value.])
          {
            [.len]
            ({[.len] [.value] (([.len] :.param) name.tensor.broadcast-bf16-at)}
              materializec)
          }
          buf.to-mem))
        "#,
    )?;
    let value = half::bf16::from_f32(3.5);

    let [Value::MemOwn(empty)] = runtime.exec(
        "tensor.broadcast-bf16-from-u16",
        [value.to_bits().into(), 0_u64.into()],
    )?
    else {
        anyhow::bail!("tensor.broadcast-bf16-from-u16 returned the wrong value kind");
    };
    assert!(empty.to_u16_vec().is_empty());

    let [Value::MemOwn(result)] = runtime.exec(
        "tensor.broadcast-bf16-from-u16",
        [value.to_bits().into(), 4_u64.into()],
    )?
    else {
        anyhow::bail!("tensor.broadcast-bf16-from-u16 returned the wrong value kind");
    };
    assert_eq!(result.to_u16_vec(), vec![value.to_bits(); 4]);
    Ok(())
}

#[test]
fn arange_bf16_test() -> anyhow::Result<()> {
    let runtime = runtime_with("")?;
    let [Value::MemOwn(empty)] = runtime.exec("tensor.arange-bf16", [0_u64.into()])? else {
        anyhow::bail!("tensor.arange-bf16 returned the wrong value kind");
    };
    assert!(empty.to_u16_vec().is_empty());

    let [Value::MemOwn(result)] = runtime.exec("tensor.arange-bf16", [5_u64.into()])? else {
        anyhow::bail!("tensor.arange-bf16 returned the wrong value kind");
    };
    assert_eq!(
        bf16_values_from_bits(&result.to_u16_vec()),
        bf16_values(&[0.0, 1.0, 2.0, 3.0, 4.0])
    );
    Ok(())
}

#[test]
fn slice_bf16_test() -> anyhow::Result<()> {
    let runtime = runtime_with("")?;
    let input_values = bf16_values(&[10.0, 20.0, 30.0, 40.0]);
    for (start, len, expected) in [
        (0_u64, 2_u64, vec![input_values[0], input_values[1]]),
        (1, 2, vec![input_values[1], input_values[2]]),
        (2, 0, vec![]),
        (0, 4, input_values.to_vec()),
    ] {
        let input = runtime.mem_u16(&bf16_bits(&input_values))?;
        let [Value::MemOwn(result)] = runtime.exec(
            "tensor.slice-bf16",
            [input.as_ref().into(), start.into(), len.into()],
        )?
        else {
            anyhow::bail!("tensor.slice-bf16 returned the wrong value kind");
        };
        assert_eq!(
            bf16_values_from_bits(&result.to_u16_vec()),
            expected,
            "slice({start}, {len})"
        );
    }
    Ok(())
}

#[test]
fn concat_bf16_test() -> anyhow::Result<()> {
    let runtime = runtime_with("")?;
    for (left, right, expected) in [
        (
            bf16_values(&[1.0, 2.0]),
            bf16_values(&[3.0, 4.0]),
            bf16_values(&[1.0, 2.0, 3.0, 4.0]),
        ),
        (
            bf16_values(&[1.0]),
            bf16_values(&[2.0, 3.0, 4.0]),
            bf16_values(&[1.0, 2.0, 3.0, 4.0]),
        ),
        (vec![], bf16_values(&[3.0, 4.0]), bf16_values(&[3.0, 4.0])),
        (bf16_values(&[1.0, 2.0]), vec![], bf16_values(&[1.0, 2.0])),
        (vec![], vec![], vec![]),
    ] {
        let left_mem = runtime.mem_u16(&bf16_bits(&left))?;
        let right_mem = runtime.mem_u16(&bf16_bits(&right))?;
        let [Value::MemOwn(result)] = runtime.exec(
            "tensor.concat-bf16",
            [left_mem.as_ref().into(), right_mem.as_ref().into()],
        )?
        else {
            anyhow::bail!("tensor.concat-bf16 returned the wrong value kind");
        };
        assert_eq!(bf16_values_from_bits(&result.to_u16_vec()), expected);
    }
    Ok(())
}

#[test]
fn argmax_bf16_test() -> anyhow::Result<()> {
    let runtime = runtime_with("")?;

    for (input_values, expected) in [
        (bf16_values(&[5.0]), 0_u64),
        (bf16_values(&[1.0, 2.0, 4.0]), 2),
        (bf16_values(&[4.0, 2.0, 1.0]), 0),
        (bf16_values(&[1.0, 7.0, 3.0, 2.0]), 1),
        (bf16_values(&[1.0, 7.0, 7.0, 2.0]), 1),
        (bf16_values(&[3.0, 3.0, 3.0]), 0),
    ] {
        let input = runtime.mem_u16(&bf16_bits(&input_values))?;
        let [result] = runtime.exec("tensor.argmax-bf16", [input.as_ref().into()])?;
        let Value::U64(result) = result else {
            anyhow::bail!("tensor.argmax-bf16 returned non-u64 value: {result:?}");
        };
        assert_eq!(result, expected, "tensor.argmax-bf16({input_values:?})");
    }

    Ok(())
}

const SUM_BF16_ENTRYPOINT: &str = r#"
(def program sum-bf16-as-u16 : (cap.ref mem) -> (u16 val) =
  (mem.cast.bf16 [len input.]
    {(const.u32.0x00000000 u32.bitcast-f32 f32.to-bf16)
     unit.intro
     name.bf16.add-env
     ({[.input] (([.len] :.param) name.bf16-at)})
     [.len]}
    reducec
    bf16.bitcast-u16))
"#;

#[test]
fn reducec_accumulates_a_long_bf16_sequence() -> anyhow::Result<()> {
    use half::bf16;

    // U16 adapts the BF16 result for use as a runtime entrypoint; the reduction
    // and every intermediate rounding step remain BF16 inside generated code.
    let runtime = runtime_with(SUM_BF16_ENTRYPOINT)?;
    let values = (0..257)
        .map(|index| {
            let value = ((index % 29) as f32 - 14.0) / 17.0;
            bf16::from_f32(value)
        })
        .collect::<Vec<_>>();
    let expected = values.iter().copied().fold(bf16::ZERO, |sum, value| {
        bf16::from_f32(sum.to_f32() + value.to_f32())
    });

    let input = runtime.mem_u16(&bf16_bits(&values))?;
    let [Value::U16(actual)] = runtime.exec("sum-bf16-as-u16", [input.as_ref().into()])? else {
        anyhow::bail!("sum-bf16-as-u16 returned the wrong value kind");
    };

    assert_eq!(actual, expected.to_bits());
    Ok(())
}
