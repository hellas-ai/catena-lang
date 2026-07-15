use catena_lang::{
    closure2::region::find_regions,
    compile::{CompileError, compile},
    pass::forget_closures::Region,
    report::CompileReport,
};
use hexpr::Operation;
use metacat::theory::{RawTheorySet, Theory, TheoryId};

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

fn compile_through_closure_conversion(source: &str) -> anyhow::Result<CompileReport> {
    let raw = RawTheorySet::from_texts(STDLIB.iter().copied().chain([source]))?;
    let report = match compile(raw) {
        Ok(report) => report,
        Err(failure) if matches!(failure.cause, CompileError::NotImplementedError) => {
            failure.report
        }
        Err(failure) => return Err(failure.into()),
    };
    anyhow::ensure!(
        report.forgotten_closures.is_some(),
        "compile stopped before forget_closures completed"
    );
    anyhow::ensure!(
        report.closure_conversion.is_some(),
        "compile stopped before closure conversion completed"
    );
    Ok(report)
}

#[test]
fn defer_bool_id() -> anyhow::Result<()> {
    compile_through_closure_conversion(
        r#"
        (def program defer-bool-id : (bool val) -> (bool val) = (
          {defer (name.bool.id lift)}
          compose
          run
        ))
        "#,
    )?;
    Ok(())
}

#[test]
fn run_named_and_packed_with_free() -> anyhow::Result<()> {
    compile_through_closure_conversion(
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
    )?;
    Ok(())
}

#[test]
fn closure2_examples_emit_expected_region_boundaries() -> anyhow::Result<()> {
    let report = compile_through_closure_conversion(include_str!("../examples/closure2.hex"))?;
    let program = TheoryId(op("program"));
    let definitions = report
        .forgotten_closures
        .as_ref()
        .and_then(|theories| theories.get(&program))
        .ok_or_else(|| anyhow::anyhow!("forget_closures did not emit the program theory"))?;

    for (definition, expected_regions) in [
        ("closure2-named-if", 2),
        ("closure2-captured-if", 2),
        ("closure2-composed-if", 2),
        ("closure2-tensored-if", 2),
        ("closure2-packed-closure", 2),
        ("closure2-reduce", 2),
    ] {
        let term = definitions
            .get(&op(definition))
            .ok_or_else(|| anyhow::anyhow!("missing forgotten definition `{definition}`"))?;
        let actual_regions = find_regions(term)?.len();
        anyhow::ensure!(
            actual_regions == expected_regions,
            "`{definition}` emitted {actual_regions} !closure edges; expected {expected_regions}"
        );

        for (edge, adjacency) in term
            .hypergraph
            .edges
            .iter()
            .zip(&term.hypergraph.adjacency)
            .filter(|(edge, _)| matches!(edge, Region::Closure))
        {
            anyhow::ensure!(
                adjacency.sources.len() == 2 && adjacency.targets.len() == 1,
                "`{definition}` emitted malformed {edge}: expected two control-flow inputs and one bracketed output"
            );
        }
    }

    Ok(())
}

#[test]
fn closure2_finds_named_and_captured_regions() -> anyhow::Result<()> {
    let report = compile_through_closure_conversion(include_str!("../examples/closure2.hex"))?;
    let definitions = report
        .forgotten_closures
        .as_ref()
        .and_then(|theories| theories.get(&TheoryId(op("program"))))
        .ok_or_else(|| anyhow::anyhow!("forget_closures did not emit the program theory"))?;

    let named = definitions
        .get(&op("closure2-named-if"))
        .ok_or_else(|| anyhow::anyhow!("missing closure2-named-if"))?;
    let named_regions = find_regions(named)?;
    anyhow::ensure!(named_regions.len() == 2);
    for region in named_regions {
        anyhow::ensure!(region.environment.is_empty());
        anyhow::ensure!(
            region.edges.len() == 2,
            "named body should contain name + eval"
        );
    }

    let captured = definitions
        .get(&op("closure2-captured-if"))
        .ok_or_else(|| anyhow::anyhow!("missing closure2-captured-if"))?;
    let captured_regions = find_regions(captured)?;
    anyhow::ensure!(captured_regions.len() == 2);
    for region in captured_regions {
        anyhow::ensure!(region.environment == vec![region.codomain]);
        anyhow::ensure!(region.edges.is_empty());
    }

    Ok(())
}

#[test]
fn closure2_builds_closure_and_name_arrows() -> anyhow::Result<()> {
    let report = compile_through_closure_conversion(include_str!("../examples/closure2.hex"))?;
    let generated = report
        .closure_conversion
        .as_ref()
        .map(|conversion| &conversion.definitions)
        .ok_or_else(|| anyhow::anyhow!("missing generated closure definitions"))?;
    let Theory::Theory { arrows, .. } = generated
        .theories
        .get(&TheoryId(op("program")))
        .ok_or_else(|| anyhow::anyhow!("missing generated program theory"))?
    else {
        anyhow::bail!("program should be a user theory");
    };

    for definition in ["closure2-named-if", "closure2-captured-if"] {
        let closure_prefix = format!("closure.{definition}.");
        let name_prefix = format!("name.{closure_prefix}");
        let closures = arrows
            .iter()
            .filter(|(name, _)| name.as_str().starts_with(&closure_prefix))
            .collect::<Vec<_>>();
        let names = arrows
            .iter()
            .filter(|(name, _)| name.as_str().starts_with(&name_prefix))
            .collect::<Vec<_>>();

        anyhow::ensure!(
            closures.len() == 2,
            "expected two closures for {definition}"
        );
        anyhow::ensure!(names.len() == 2, "expected two names for {definition}");
        for (_, closure) in closures {
            anyhow::ensure!(closure.definition.is_some());
            anyhow::ensure!(closure.type_maps.0.targets.len() == 2);
            anyhow::ensure!(closure.type_maps.1.targets.len() == 1);
        }
        for (_, name) in names {
            anyhow::ensure!(name.definition.is_none());
            anyhow::ensure!(name.type_maps.1.targets.len() == 1);
        }
    }

    catena_lang::check::check(generated)?;
    Ok(())
}

#[test]
fn closure2_replacement_uses_generated_name_boundaries() -> anyhow::Result<()> {
    let report = compile_through_closure_conversion(include_str!("../examples/closure2.hex"))?;
    let theory_id = TheoryId(op("program"));
    let definitions = report
        .closure_conversion
        .as_ref()
        .map(|conversion| &conversion.replacements)
        .and_then(|theories| theories.get(&theory_id))
        .ok_or_else(|| anyhow::anyhow!("missing replaced program definitions"))?;
    let Theory::Theory { arrows, .. } = report
        .closure_conversion
        .as_ref()
        .map(|conversion| &conversion.theory_set)
        .and_then(|theories| theories.theories.get(&theory_id))
        .ok_or_else(|| anyhow::anyhow!("missing replaced program theory"))?
    else {
        anyhow::bail!("program should be a user theory");
    };

    for definition in ["closure2-named-if", "closure2-captured-if"] {
        let term = definitions
            .get(&op(definition))
            .ok_or_else(|| anyhow::anyhow!("missing replacement for {definition}"))?;
        anyhow::ensure!(
            term.hypergraph
                .edges
                .iter()
                .any(|op| op.as_str() == "bool.ifc")
        );
        anyhow::ensure!(
            !term
                .hypergraph
                .edges
                .iter()
                .any(|op| op.as_str() == "bool.if")
        );

        for (context, boundary) in term
            .hypergraph
            .edges
            .iter()
            .zip(&term.hypergraph.adjacency)
            .filter(|(op, _)| op.as_str().starts_with("__catena_context.closure."))
        {
            let name: Operation = context
                .as_str()
                .strip_prefix("__catena_context.")
                .map(|closure| format!("name.{closure}"))
                .expect("generated context prefix should strip")
                .parse()?;
            let name_arrow = arrows
                .get(&name)
                .ok_or_else(|| anyhow::anyhow!("missing wired name declaration {name}"))?;
            anyhow::ensure!(
                boundary.targets.len()
                    == boundary.sources.len() + name_arrow.type_maps.0.targets.len(),
                "context outputs should be environment copies followed by the actual name inputs"
            );
        }
    }
    Ok(())
}

fn op(name: &str) -> Operation {
    name.parse().expect("test operation should parse")
}
