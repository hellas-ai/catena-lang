use catena_lang::elaborate::{ElaborateError, elaborate};
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

    assert!(
        matches!(
            elaborate(raw),
            Err(ElaborateError::TypeMapDomainMismatch {
                theory,
                arrow,
                source_domain,
                target_domain,
            }) if theory == "program"
                && arrow == "bad"
                && source_domain == "1"
                && target_domain == "0"
        ),
        "elaboration should reject invalid arrow domains before generating name.* arrows"
    );
}
