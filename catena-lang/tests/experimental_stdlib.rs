use catena_lang::{check, elaborate, stdlib::BundleRegistry};
use metacat::theory::{RawTheorySet, TheorySet};

#[test]
fn experimental_matmul_fixtures_compile_without_gpu() -> anyhow::Result<()> {
    use catena_lang::{
        codegen::CodegenKind, compile::compile, report::CompileReport, runtime::GpuDialect,
    };

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut registry = BundleRegistry::new(&[])?;
    let name = registry.add_directory(root.join("stdlib/experimental"))?;
    let files = registry.resolve(&[&name])?;
    for fixture in ["matmul_naive_u64.hex", "matmul_tiled_u64.hex"] {
        let source = std::fs::read_to_string(
            root.join("tests/experimental_runtime/fixtures")
                .join(fixture),
        )?;
        let raw = RawTheorySet::from_texts(
            files
                .iter()
                .map(|file| file.source.as_ref())
                .chain([source.as_str()]),
        )?;
        let mut report = CompileReport::new(raw);
        let module = compile(&mut report, CodegenKind::Experimental, GpuDialect::Hip)
            .unwrap_or_else(|error| panic!("compiling {fixture}: {error}"));
        let entry = if fixture == "matmul_naive_u64.hex" {
            "matmul_simple_naive.main.matmul"
        } else {
            "matmul_simple_tiled.main.matmul"
        };
        assert!(
            module
                .functions
                .iter()
                .any(|function| function.source_name == entry)
        );
    }
    Ok(())
}

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

#[test]
fn embedded_experimental_sources_match_the_manifest() -> anyhow::Result<()> {
    let mut registry = BundleRegistry::new(&[])?;
    let name = registry.add_directory(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("stdlib/experimental"),
    )?;
    let files = registry.resolve(&[&name])?;
    assert_eq!(
        catena_lang::stdlib::experimental_sources().collect::<Vec<_>>(),
        files
            .iter()
            .map(|file| file.source.as_ref())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        catena_lang::stdlib::experimental_core_sources().collect::<Vec<_>>(),
        files
            .iter()
            .filter(|file| file
                .filename
                .file_name()
                .is_none_or(|name| name != "matmul.hex"))
            .map(|file| file.source.as_ref())
            .collect::<Vec<_>>()
    );
    Ok(())
}
