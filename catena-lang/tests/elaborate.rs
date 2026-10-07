use catena_lang::elaborate::{ElaborateError, elaborate};
use metacat::theory::RawTheorySet;

#[test]
fn missing_type_literals_are_available_before_name_generation() -> anyhow::Result<()> {
    let source = "(theory program type {
        (arr test.literal : ([] {}) -> ([] {({type.const.u64.0x0000000000000080 u64} :)}))
    })";
    let raw = RawTheorySet::from_texts(catena_lang::stdlib::sources().chain([source]))?;
    let elaborated = elaborate(raw)?;
    assert!(
        elaborated.theories[&"type".parse()?]
            .arrows
            .contains_key(&"type.const.u64.0x0000000000000080".parse()?)
    );
    metacat::theory::TheorySet::from_raw(elaborated)?;
    Ok(())
}

#[test]
fn program_literals_retain_symbolic_identity_without_declarations() -> anyhow::Result<()> {
    for (ty, literal) in [
        ("u32", "type.const.u32.0x00000001"),
        ("u64", "type.const.u64.0x0000000000000001"),
    ] {
        let source = format!(
            "(theory program type {{
            (def test.literal : ([] {{}}) -> ([] {{({ty} val)}})
                = ({literal} :.forget))
        }})"
        );
        let raw = RawTheorySet::from_texts(catena_lang::stdlib::sources())?
            .merge(RawTheorySet::from_text(&source)?)?;
        let elaborated = elaborate(raw)?;
        let program = &elaborated.theories[&"program".parse()?];
        let arrow = &program.arrows[&literal.parse()?];
        assert_eq!(arrow.type_maps.0, "([] {})".parse()?);
        assert_eq!(
            arrow.type_maps.1,
            format!("([] {{({{{literal} {ty}}} :)}})").parse()?
        );
        assert!(arrow.definition.is_none());
        assert!(
            program
                .arrows
                .contains_key(&format!("name.{literal}").parse()?)
        );
        let theories = metacat::theory::TheorySet::from_raw(elaborated)?;
        catena_lang::check::check(&theories)?;
    }
    Ok(())
}

#[test]
fn conflicting_program_literal_declarations_are_rejected() -> anyhow::Result<()> {
    let literal = "type.const.u64.0x0000000000000001";
    for declaration in [
        format!("(arr {literal} : ([] {{}}) -> ([] {{(u64 val)}}))"),
        format!("(arr {literal} : ([] {{(u64 val)}}) -> ([] {{({{{literal} u64}} :)}}))"),
        format!("(arr {literal} : ([] {{}}) -> ([] {{({{{literal} u32}} :)}}))"),
        format!("(def {literal} : ([] {{}}) -> ([] {{({{{literal} u64}} :)}}) = [])"),
    ] {
        let source = format!(
            "(theory program type {{
            {declaration}
            (arr test.literal : ([] {{}}) -> ([] {{({{{literal} u64}} :)}}))
        }})"
        );
        let raw = RawTheorySet::from_texts(catena_lang::stdlib::sources())?
            .merge(RawTheorySet::from_text(&source)?)?;
        assert!(
            matches!(elaborate(raw), Err(ElaborateError::InvalidConstant { operation, .. }) if operation == literal)
        );
    }
    Ok(())
}

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
