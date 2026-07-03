use catena_lang::{compile::compile, stdlib};
use metacat::theory::RawTheorySet;

#[test]
fn rejects_arrow_type_maps_with_different_context_domains() -> anyhow::Result<()> {
    let source = r#"
        (def program bad :
          ({[n] u64} :)
          ->
          (u64 val)
        = ([n.] u64.one))
        "#;
    let raw = RawTheorySet::from_texts(stdlib::sources().chain([source]))?;

    let error = compile(raw).expect_err("arrow type maps with different domains should fail");
    let message = error.to_string();
    assert!(
        message.contains(
            "arrow `program.bad` source and target type maps must have the same context domain"
        ),
        "unexpected error message: {message}"
    );

    Ok(())
}
