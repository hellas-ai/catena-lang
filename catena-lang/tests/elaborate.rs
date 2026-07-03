use catena_lang::elaborate::elaborate;
use metacat::theory::RawTheorySet;

#[test]
fn rejects_arrow_type_maps_with_different_context_domains_before_name_generation() {
    let raw = RawTheorySet::from_text(
        r#"
        (theory type nat {
          (arr : : 2 -> 1)
          (arr val : 1 -> 1)
          (arr u64 : 0 -> 1)
        })

        (theory program type {
          (arr bad :
            ({[n] u64} :)
            ->
            (u64 val))
        })
        "#,
    )
    .expect("test theory should parse");

    let Err(error) = elaborate(raw) else {
        panic!("elaboration should reject invalid arrow domains before generating name.* arrows");
    };
    let message = error.to_string();
    assert!(
        message.contains("source and target type maps must have the same context domain"),
        "unexpected error message: {message}"
    );
}
