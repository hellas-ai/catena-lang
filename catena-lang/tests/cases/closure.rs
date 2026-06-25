use super::*;

const SOURCE: &str = include_str!("closure/basic.hex");

#[test]
#[ignore = "depends on symbol resolution bug"]
fn if_id_neg_test() -> anyhow::Result<()> {
    let runtime = runtime_with(SOURCE)?;
    let input = 1.0f32;
    let [result] = runtime.exec("if-id-neg", [false.into(), input.into()])?;

    let Value::F32(result) = result else {
        anyhow::bail!("log-approx returned non-f32 value: {result:?}");
    };

    let expected = -input;
    assert_eq!(expected, result);
    Ok(())
}
