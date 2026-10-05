use catena_lang::{check, elaborate, stdlib::BundleRegistry};
use metacat::theory::{RawTheorySet, TheorySet};

#[test]
fn experimental_library_typechecks_without_default_sources() -> anyhow::Result<()> {
    let mut registry = BundleRegistry::new(&[])?;
    let name = registry.add_directory(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("stdlib/experimental"),
    )?;
    let files = registry.resolve(&[&name])?;
    let raw = RawTheorySet::from_texts(files.iter().map(|file| file.source.as_ref()))?;
    let types = &raw.theories[&"type".parse()?].arrows;
    let mut referenced = std::collections::BTreeSet::new();
    for arrow in raw.theories[&"program".parse()?].arrows.values() {
        collect_operations(&arrow.type_maps.0, &mut referenced);
        collect_operations(&arrow.type_maps.1, &mut referenced);
    }
    let missing = referenced
        .into_iter()
        .filter(|op| !types.contains_key(op))
        .collect::<Vec<_>>();
    assert!(
        missing.is_empty(),
        "undeclared foundational types: {missing:?}"
    );
    let elaborated = elaborate::elaborate(raw)?;
    let mut referenced = std::collections::BTreeSet::new();
    let program = &elaborated.theories[&"program".parse()?].arrows;
    for arrow in program.values() {
        if let Some(definition) = &arrow.definition {
            collect_operations(definition, &mut referenced);
        }
    }
    let missing = referenced
        .into_iter()
        .filter(|op| !program.contains_key(op))
        .collect::<Vec<_>>();
    assert!(
        missing.is_empty(),
        "undeclared foundational operations: {missing:?}"
    );
    let theories = TheorySet::from_raw(elaborated)?;
    check::check(&theories)?;
    Ok(())
}

fn collect_operations(
    expr: &hexpr::Hexpr,
    output: &mut std::collections::BTreeSet<hexpr::Operation>,
) {
    match expr {
        hexpr::Hexpr::Operation(op) => {
            output.insert(op.clone());
        }
        hexpr::Hexpr::Composition(parts) | hexpr::Hexpr::Tensor(parts) => {
            for part in parts {
                collect_operations(part, output);
            }
        }
        _ => {}
    }
}
