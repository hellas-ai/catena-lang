use super::*;

const SOURCE: &str = include_str!("../../examples/materializec.hex");

#[test]
fn examples_compile() -> anyhow::Result<()> {
    let _runtime = runtime_with(SOURCE)?;
    Ok(())
}
