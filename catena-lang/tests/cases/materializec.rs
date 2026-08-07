use super::*;

const SOURCE: &str = include_str!("../../examples/materializec.hex");

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
