use super::*;

const SOURCE: &str = include_str!("../../examples/materializec.hex");

#[test]
fn materialize_indexes_exec() -> anyhow::Result<()> {
    let runtime = runtime_with(SOURCE)?;

    let [result] = runtime.exec("materialize-indexes", [4_u64.into()])?;
    let Value::Mem(result) = result else {
        anyhow::bail!("materialize-indexes returned non-mem value: {result:?}");
    };

    assert_eq!(result.to_u64_vec(), vec![1, 1, 1, 1]);
    Ok(())
}

#[test]
fn materialize_copy_u64_exec() -> anyhow::Result<()> {
    let runtime = runtime_with(SOURCE)?;

    let input = runtime.mem_u64(&[3, 5, 8, 13])?;
    let [result] = runtime.exec("materialize-copy-u64", [input])?;
    let Value::Mem(result) = result else {
        anyhow::bail!("materialize-copy-u64 returned non-mem value: {result:?}");
    };

    assert_eq!(result.to_u64_vec(), vec![3, 5, 8, 13]);
    Ok(())
}
