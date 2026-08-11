use super::*;

const SOURCE: &str = include_str!("../../examples/materializec.hex");

fn canonical_adjacent_f32_sum(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut terms = values.to_vec();
    let mut stride = 1;
    while stride < terms.len() {
        let step = stride * 2;
        for left in (0..terms.len().saturating_sub(stride)).step_by(step) {
            terms[left] += terms[left + stride];
        }
        stride *= 2;
    }
    terms[0]
}

#[test]
fn materialize_indexes_exec() -> anyhow::Result<()> {
    let runtime = runtime_with(SOURCE)?;

    let [result] = runtime.exec("materialize-indexes", [4_u64.into()])?;
    let Value::MemOwn(result) = result else {
        anyhow::bail!("materialize-indexes returned non-mem value: {result:?}");
    };

    assert_eq!(result.to_u64_vec(), vec![1, 1, 1, 1]);
    Ok(())
}

#[test]
fn materialize_copy_u64_exec() -> anyhow::Result<()> {
    let runtime = runtime_with(SOURCE)?;

    let input = runtime.mem_u64(&[3, 5, 8, 13])?;
    let [result] = runtime.exec("materialize-copy-u64", [input.as_ref().into()])?;
    let Value::MemOwn(result) = result else {
        anyhow::bail!("materialize-copy-u64 returned non-mem value: {result:?}");
    };

    assert_eq!(result.to_u64_vec(), vec![3, 5, 8, 13]);
    Ok(())
}

#[test]
fn materialize_into_updates_only_the_selected_owned_range() -> anyhow::Result<()> {
    let runtime = runtime_with(SOURCE)?;
    let input = runtime.mem_f32(&[9.0, 9.0, 9.0, 9.0])?;

    let [result] = runtime.exec(
        "materialize-into-f32",
        [input.into(), 1_u64.into(), 2_u64.into()],
    )?;
    let Value::MemOwn(result) = result else {
        anyhow::bail!("materialize-into-f32 returned non-mem value: {result:?}");
    };

    assert_eq!(result.to_f32_vec(), vec![9.0, 0.0, 1.0, 9.0]);
    Ok(())
}

#[test]
fn materialize_borrow_returns_source_owner_and_new_output() -> anyhow::Result<()> {
    let runtime = runtime_with(SOURCE)?;
    let input = runtime.mem_f32(&[3.0, 5.0, 8.0, 13.0])?;
    let input_ptr = input.as_ptr();

    let [source, copied] = runtime.exec("materialize-borrow-f32", [input.into(), 3_u64.into()])?;
    let Value::MemOwn(source) = source else {
        anyhow::bail!("materialize-borrow-f32 returned non-mem source: {source:?}");
    };
    let Value::MemOwn(copied) = copied else {
        anyhow::bail!("materialize-borrow-f32 returned non-mem output: {copied:?}");
    };

    assert_eq!(source.as_ptr(), input_ptr);
    assert_eq!(source.to_f32_vec(), vec![3.0, 5.0, 8.0, 13.0]);
    assert_eq!(copied.to_f32_vec(), vec![3.0, 5.0, 8.0]);
    Ok(())
}

#[test]
fn materialize_reduce_f32_uses_one_cooperative_reduction_per_output() -> anyhow::Result<()> {
    let runtime = runtime_with(SOURCE)?;
    let input = runtime.mem_f32(&[1.0, 2.0, 3.0, 5.0, 8.0, 13.0])?;

    let [output] = runtime.exec(
        "materialize-reduce-f32-rows",
        [input.as_ref().into(), 2_u64.into(), 3_u64.into()],
    )?;
    let Value::MemOwn(output) = output else {
        anyhow::bail!("materialize-reduce-f32-rows returned non-memory: {output:?}");
    };

    assert_eq!(output.to_f32_vec(), vec![6.0, 26.0]);
    Ok(())
}

#[test]
fn materialize_reduce_f32_uses_the_specified_logical_index_tree() -> anyhow::Result<()> {
    let runtime = runtime_with(SOURCE)?;
    let input = runtime.mem_f32(&[1.0e20, 1.0, -1.0e20, 1.0])?;

    let [output] = runtime.exec(
        "materialize-reduce-f32-rows",
        [input.as_ref().into(), 1_u64.into(), 4_u64.into()],
    )?;
    let Value::MemOwn(output) = output else {
        anyhow::bail!("materialize-reduce-f32-rows returned non-memory: {output:?}");
    };

    // The specified tree is (term(0) + term(1)) + (term(2) + term(3)).
    // A lane-oriented tree would instead produce 2.0 for these values.
    assert_eq!(output.to_f32_vec()[0].to_bits(), 0.0_f32.to_bits());
    Ok(())
}

#[test]
fn materialize_reduce_f32_empty_reduction_is_positive_zero() -> anyhow::Result<()> {
    let runtime = runtime_with(SOURCE)?;
    let unused_input = runtime.mem_f32(&[42.0])?;

    let [output] = runtime.exec(
        "materialize-reduce-f32-rows",
        [unused_input.as_ref().into(), 1_u64.into(), 0_u64.into()],
    )?;
    let Value::MemOwn(output) = output else {
        anyhow::bail!("materialize-reduce-f32-rows returned non-memory: {output:?}");
    };

    assert_eq!(output.to_f32_vec()[0].to_bits(), 0.0_f32.to_bits());
    Ok(())
}

#[test]
fn materialize_reduce_f32_pair_shares_barriers_without_changing_results() -> anyhow::Result<()> {
    let runtime = runtime_with(SOURCE)?;
    let input = runtime.mem_f32(&[1.0, 2.0, 3.0, 5.0, 8.0, 13.0])?;

    let [left, right] = runtime.exec(
        "materialize-reduce-f32-pair-rows",
        [input.as_ref().into(), 2_u64.into(), 3_u64.into()],
    )?;
    let Value::MemOwn(left) = left else {
        anyhow::bail!("reduce-f32-pair returned non-memory left output: {left:?}");
    };
    let Value::MemOwn(right) = right else {
        anyhow::bail!("reduce-f32-pair returned non-memory right output: {right:?}");
    };

    assert_eq!(left.to_f32_vec(), vec![6.0, 26.0]);
    assert_eq!(right.to_f32_vec(), vec![6.0, 26.0]);
    Ok(())
}

#[test]
fn materialize_reduce_f32_pair_preserves_both_logical_index_trees() -> anyhow::Result<()> {
    let runtime = runtime_with(SOURCE)?;
    let input = runtime.mem_f32(&[1.0e20, 1.0, -1.0e20, 1.0])?;

    let [left, right] = runtime.exec(
        "materialize-reduce-f32-pair-rows",
        [input.as_ref().into(), 1_u64.into(), 4_u64.into()],
    )?;
    for output in [left, right] {
        let Value::MemOwn(output) = output else {
            anyhow::bail!("reduce-f32-pair returned non-memory output: {output:?}");
        };
        assert_eq!(output.to_f32_vec()[0].to_bits(), 0.0_f32.to_bits());
    }
    Ok(())
}

#[test]
fn materialize_reduction_wave_tail_matches_canonical_bits() -> anyhow::Result<()> {
    let runtime = runtime_with(SOURCE)?;

    for len in [257_usize, 2_048, 6_144] {
        let values = (0..len)
            .map(|index| match index % 8 {
                0 => 1.0e20,
                1 => 1.0,
                2 => -1.0e20,
                3 => 1.0,
                4 => -0.0,
                5 => 0.25,
                6 => -0.5,
                _ => 0.125,
            })
            .collect::<Vec<_>>();
        let expected = canonical_adjacent_f32_sum(&values).to_bits();

        if len <= 4_096 {
            let input = runtime.mem_f32(&values)?;
            let [output] = runtime.exec(
                "materialize-reduce-f32-rows",
                [input.as_ref().into(), 1_u64.into(), (len as u64).into()],
            )?;
            let Value::MemOwn(output) = output else {
                anyhow::bail!("reduce-f32 returned non-memory output: {output:?}");
            };
            assert_eq!(output.to_f32_vec()[0].to_bits(), expected, "len {len}");

            let [left, right] = runtime.exec(
                "materialize-reduce-f32-pair-rows",
                [input.as_ref().into(), 1_u64.into(), (len as u64).into()],
            )?;
            for output in [left, right] {
                let Value::MemOwn(output) = output else {
                    anyhow::bail!("reduce-f32-pair returned non-memory output: {output:?}");
                };
                assert_eq!(output.to_f32_vec()[0].to_bits(), expected, "len {len}");
            }
        }

        let input = runtime.mem_f32(&values)?;
        let [_source, output] = runtime.exec(
            "materialize-borrow-reduce-f32-rows",
            [input.into(), 1_u64.into(), (len as u64).into()],
        )?;
        let Value::MemOwn(output) = output else {
            anyhow::bail!("borrow-reduce-f32 returned non-memory output: {output:?}");
        };
        assert_eq!(output.to_f32_vec()[0].to_bits(), expected, "len {len}");
    }
    Ok(())
}

#[test]
fn materialize_borrow_routed_bf16_gemv_pair_executes_both_trees() -> anyhow::Result<()> {
    let runtime = runtime_with(SOURCE)?;
    let input_values = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -3.0];
    let input = runtime.mem_f32(&input_values)?;
    let input_ptr = input.as_ptr();
    let selected = runtime.mem_u64(&[1, 0])?;
    let selected_ptr = selected.as_ptr();
    let gate_values = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let up_values = [
        1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0,
    ];
    let gate_bits = gate_values
        .into_iter()
        .map(|value| half::bf16::from_f32(value).to_bits())
        .collect::<Vec<_>>();
    let up_bits = up_values
        .into_iter()
        .map(|value| half::bf16::from_f32(value).to_bits())
        .collect::<Vec<_>>();
    let gate = runtime.mem_u16(&gate_bits)?;
    let up = runtime.mem_u16(&up_bits)?;

    let [input_after, selected_after, gate_output, up_output] = runtime.exec(
        "materialize-borrow-routed-bf16-gemv-pair",
        [
            input.into(),
            gate.as_ref().into(),
            up.as_ref().into(),
            selected.into(),
        ],
    )?;
    let Value::MemOwn(input_after) = input_after else {
        anyhow::bail!("routed pair returned non-memory input owner: {input_after:?}");
    };
    let Value::MemOwn(selected_after) = selected_after else {
        anyhow::bail!("routed pair returned non-memory selected owner: {selected_after:?}");
    };
    let Value::MemOwn(gate_output) = gate_output else {
        anyhow::bail!("routed pair returned non-memory gate output: {gate_output:?}");
    };
    let Value::MemOwn(up_output) = up_output else {
        anyhow::bail!("routed pair returned non-memory up output: {up_output:?}");
    };

    assert_eq!(input_after.as_ptr(), input_ptr);
    assert_eq!(selected_after.as_ptr(), selected_ptr);
    assert_eq!(input_after.to_f32_vec(), input_values);
    assert_eq!(selected_after.to_u64_vec(), vec![1, 0]);
    assert_eq!(gate_output.to_f32_vec(), vec![3.0, 4.0, -1.0, 0.5]);
    assert_eq!(up_output.to_f32_vec(), vec![2.0, 4.0, -1.5, 3.5]);
    Ok(())
}

#[test]
fn materialize_softmax_f32_normalizes_rows_in_place() -> anyhow::Result<()> {
    let runtime = runtime_with(SOURCE)?;
    let input = runtime.mem_f32(&[0.0, 1.0, 2.0, f32::NEG_INFINITY, 3.0, 3.0])?;
    let input_ptr = input.as_ptr();

    let [output] = runtime.exec("materialize-softmax-f32-rows", [input.into(), 3_u64.into()])?;
    let Value::MemOwn(output) = output else {
        anyhow::bail!("materialize-softmax-f32-rows returned non-memory: {output:?}");
    };

    assert_eq!(output.as_ptr(), input_ptr);
    assert_eq!(
        output.to_f32_vec(),
        vec![0.0, 1.0 / 3.0, 2.0 / 3.0, 0.0, 0.5, 0.5]
    );
    Ok(())
}

#[test]
fn materialize_borrow_reduce_f32_returns_source_and_reductions() -> anyhow::Result<()> {
    let runtime = runtime_with(SOURCE)?;
    let input = runtime.mem_f32(&[1.0, 2.0, 3.0, 5.0, 8.0, 13.0])?;
    let input_ptr = input.as_ptr();

    let [source, output] = runtime.exec(
        "materialize-borrow-reduce-f32-rows",
        [input.into(), 2_u64.into(), 3_u64.into()],
    )?;
    let Value::MemOwn(source) = source else {
        anyhow::bail!("borrow-reduce returned non-memory source: {source:?}");
    };
    let Value::MemOwn(output) = output else {
        anyhow::bail!("borrow-reduce returned non-memory output: {output:?}");
    };

    assert_eq!(source.as_ptr(), input_ptr);
    assert_eq!(source.to_f32_vec(), vec![1.0, 2.0, 3.0, 5.0, 8.0, 13.0]);
    assert_eq!(output.to_f32_vec(), vec![6.0, 26.0]);
    Ok(())
}

#[test]
fn materialize_borrow_reduce_f32_preserves_the_specified_tree() -> anyhow::Result<()> {
    let runtime = runtime_with(SOURCE)?;
    let input = runtime.mem_f32(&[1.0e20, 1.0, -1.0e20, 1.0])?;

    let [_source, output] = runtime.exec(
        "materialize-borrow-reduce-f32-rows",
        [input.into(), 1_u64.into(), 4_u64.into()],
    )?;
    let Value::MemOwn(output) = output else {
        anyhow::bail!("borrow-reduce returned non-memory output: {output:?}");
    };

    assert_eq!(output.to_f32_vec()[0].to_bits(), 0.0_f32.to_bits());
    Ok(())
}

#[test]
fn materialize_borrow_argmax_f32_returns_source_and_last_maximum() -> anyhow::Result<()> {
    let runtime = runtime_with(SOURCE)?;
    let input = runtime.mem_f32(&[3.0, 7.0, -1.0, 7.0])?;
    let input_ptr = input.as_ptr();

    let [source, output] = runtime.exec("materialize-borrow-argmax-f32", [input.into()])?;
    let Value::MemOwn(source) = source else {
        anyhow::bail!("borrow-argmax returned non-memory source: {source:?}");
    };
    let Value::MemOwn(output) = output else {
        anyhow::bail!("borrow-argmax returned non-memory output: {output:?}");
    };

    assert_eq!(source.as_ptr(), input_ptr);
    assert_eq!(source.to_f32_vec(), vec![3.0, 7.0, -1.0, 7.0]);
    assert_eq!(output.to_u64_vec(), vec![3]);
    Ok(())
}

#[test]
fn materialize_borrow_topk_f32_is_stable_and_canonicalizes_zero() -> anyhow::Result<()> {
    let runtime = runtime_with(SOURCE)?;
    let values = [
        3.0,
        7.0,
        -1.0,
        7.0,
        5.0,
        9.0,
        9.0,
        0.0,
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.0,
        -1.0,
        1.0,
        0.0,
    ];
    let input = runtime.mem_f32(&values)?;
    let input_ptr = input.as_ptr();

    let [source, output] = runtime.exec("materialize-borrow-topk-f32", [input.into()])?;
    let Value::MemOwn(source) = source else {
        anyhow::bail!("borrow-topk returned non-memory source: {source:?}");
    };
    let Value::MemOwn(output) = output else {
        anyhow::bail!("borrow-topk returned non-memory output: {output:?}");
    };

    assert_eq!(source.as_ptr(), input_ptr);
    assert_eq!(
        source
            .to_f32_vec()
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        values.into_iter().map(f32::to_bits).collect::<Vec<_>>()
    );
    assert_eq!(
        output.to_u64_vec(),
        vec![5, 6, 1, 3, 4, 0, 7, 2, 2, 4, 6, 0, 1, 7, 5, 3]
    );
    Ok(())
}

#[test]
fn materialize_borrow_topk_f32_parallel_path_matches_stable_key_order() -> anyhow::Result<()> {
    let runtime = runtime_with(SOURCE)?;

    for columns in [128_usize, 129] {
        let values = (0..columns)
            .map(|index| match index % 11 {
                0 => -0.0,
                1 => 0.0,
                2 => 7.0,
                3 => 7.0,
                4 => f32::INFINITY,
                5 => f32::NEG_INFINITY,
                _ => (index as f32 % 17.0) - 8.0,
            })
            .collect::<Vec<_>>();
        let mut expected = (0..columns).collect::<Vec<_>>();
        expected.sort_by(|&left, &right| {
            let key = |value: f32| {
                let mut bits = value.to_bits();
                if bits & 0x7FFF_FFFF == 0 {
                    bits = 0;
                }
                if bits & 0x8000_0000 != 0 {
                    !bits
                } else {
                    bits ^ 0x8000_0000
                }
            };
            key(values[right])
                .cmp(&key(values[left]))
                .then_with(|| left.cmp(&right))
        });
        expected.truncate(8);

        let input = runtime.mem_f32(&values)?;
        let [source, output] = runtime.exec(
            "materialize-borrow-topk-f32-dynamic",
            [
                input.into(),
                1_u64.into(),
                (columns as u64).into(),
                8_u64.into(),
            ],
        )?;
        let Value::MemOwn(source) = source else {
            anyhow::bail!("dynamic top-k returned non-memory source: {source:?}");
        };
        let Value::MemOwn(output) = output else {
            anyhow::bail!("dynamic top-k returned non-memory output: {output:?}");
        };

        assert_eq!(
            source
                .to_f32_vec()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            values.iter().copied().map(f32::to_bits).collect::<Vec<_>>()
        );
        assert_eq!(
            output.to_u64_vec(),
            expected
                .into_iter()
                .map(|index| index as u64)
                .collect::<Vec<_>>(),
            "column count {columns}"
        );
    }
    Ok(())
}
