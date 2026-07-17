use super::*;

const SOURCE: &str = include_str!("materialize/basic.hex");

#[test]
fn materialize_empty_source_exec() -> anyhow::Result<()> {
    let runtime = runtime_with(SOURCE)?;

    let [result] = runtime.exec("materialize-indexes-source", [0_u64.into()])?;
    let Value::Mem(result) = result else {
        anyhow::bail!("materialize-indexes-source returned non-mem value: {result:?}");
    };

    assert_eq!(result.to_u64_vec(), Vec::<u64>::new());
    Ok(())
}

#[test]
fn materialize_indexes_source_exec() -> anyhow::Result<()> {
    let runtime = runtime_with(SOURCE)?;

    let [result] = runtime.exec("materialize-indexes-source", [4_u64.into()])?;
    let Value::Mem(result) = result else {
        anyhow::bail!("materialize-indexes-source returned non-mem value: {result:?}");
    };

    assert_eq!(result.to_u64_vec(), vec![0, 1, 2, 3]);
    Ok(())
}
