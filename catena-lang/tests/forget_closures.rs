use catena_lang::{
    closure2::region::find_regions,
    compile::{CompileError, compile},
    pass::forget_closures::Region,
    report::CompileReport,
};
use hexpr::Operation;
use metacat::theory::{RawTheorySet, TheoryId};

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

fn compile_through_forget_closures(source: &str) -> anyhow::Result<CompileReport> {
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
        report.closure_regions.is_some(),
        "compile stopped before closure region discovery completed"
    );
    Ok(report)
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
    )?;
    Ok(())
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
    )?;
    Ok(())
}

#[test]
fn closure2_examples_emit_expected_region_boundaries() -> anyhow::Result<()> {
    let report = compile_through_forget_closures(include_str!("../examples/closure2.hex"))?;
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
    let report = compile_through_forget_closures(include_str!("../examples/closure2.hex"))?;
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

fn op(name: &str) -> Operation {
    name.parse().expect("test operation should parse")
}
