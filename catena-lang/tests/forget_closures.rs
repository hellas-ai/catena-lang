use catena_lang::compile::{CompileError, compile};
use metacat::theory::RawTheorySet;

const STDLIB: &[&str] = &[
    include_str!("../stdlib/cmc.hex"),
    include_str!("../stdlib/value.hex"),
    include_str!("../stdlib/buf.hex"),
    include_str!("../stdlib/index.hex"),
    include_str!("../stdlib/data.hex"),
    include_str!("../stdlib/fn.hex"),
    include_str!("../stdlib/combinators.hex"),
    include_str!("../stdlib/product.hex"),
    include_str!("../stdlib/gpu.hex"),
];

fn compile_through_forget_closures(source: &str) -> anyhow::Result<()> {
    let raw = RawTheorySet::from_texts(STDLIB.iter().copied().chain([source]))?;
    let report = compile(raw)?;
    anyhow::ensure!(
        report.forgotten_closures.is_some(),
        "compile stopped before forget_closures completed"
    );
    Ok(())
}

#[test]
fn defer_bool_id() -> anyhow::Result<()> {
    compile_through_forget_closures(
        r#"
        (def program defer-bool-id : (bool val) -> (bool val) = (
          {defer (name.bool.id lift)}
          compose
          run
        ))
        "#,
    )
}

#[test]
fn run_named_and_packed_with_free() -> anyhow::Result<()> {
    compile_through_forget_closures(
        r#"
        (def program and-packed-with-free :
          {({(bool val) (bool val)} *) (bool val)} -> (bool val) = (
          [packed free.]
          {([.packed] *.elim) [.free]}
          {bool.and [free]}
          bool.and
        ))

        (def program run-named-and-packed-with-free :
          {({(bool val) (bool val)} *) (bool val)} -> (bool val) = (
          {(*.intro defer) (name.and-packed-with-free lift)}
          compose
          run
        ))
        "#,
    )
}
