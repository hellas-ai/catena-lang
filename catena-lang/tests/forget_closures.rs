use std::collections::{BTreeMap, BTreeSet};

use catena_lang::{
    closure2::{self, region::find_regions},
    codegen::{GpuDialect, GpuValue, gpu::render_module},
    compile::compile,
    pass::{
        forget_closures::ClosureForgotten, inline_definitions,
        record_boundary_sizes::OperationWithBoundarySizes,
    },
    report::CompileReport,
};
use hexpr::Operation;
use metacat::{
    theory::{RawTheorySet, Theory, TheoryId, TheorySet},
    tree::Tree,
};
use open_hypergraphs::lax::OpenHypergraph;

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
    let report = compile(raw)?;
    anyhow::ensure!(
        report.closure_conversion.is_some(),
        "compile stopped before closure conversion completed"
    );
    anyhow::ensure!(report.boundary_sizes.is_some());
    anyhow::ensure!(report.unpacked_products.is_some());
    anyhow::ensure!(report.gpu_modules.is_some());
    Ok(report)
}

#[test]
#[should_panic(
    expected = "closure conversion requires closure-boundary definitions to be inlined first"
)]
fn closure2_rejects_uninlined_closure_boundary_definitions() {
    let raw = RawTheorySet::from_texts(STDLIB.iter().copied().chain([r#"
        (def program returns-closure :
          (bool val) -> ({1 (bool val)} =>)
        = (
          [captured.]
          ([.captured] defer)
        ))
    "#]))
    .expect("test theories should parse");
    let elaborated = catena_lang::elaborate::elaborate(raw).expect("test theory should elaborate");
    let theory_set = TheorySet::from_raw(elaborated).expect("test theory should interpret");

    closure2::run(&theory_set, &BTreeMap::new())
        .expect("closure2 should not recover from skipped boundary inlining");
}

#[test]
fn closure_boundary_definition_is_retained_for_its_name_after_direct_calls_inline()
-> anyhow::Result<()> {
    let named_use = r#"
        (def program closure2-nested-named-retention :
          {(bool val) (bool val) (bool val)} -> (bool val)
        = ([lhs rhs flag.]
          ({
            ({([.lhs] defer) ([.rhs] defer)} *.intro)
            [.flag]
          } *.intro [input.])
          {([.input] defer) (name.closure2.consume-nested lift)}
          compose
          run
        ))
    "#;
    let raw = RawTheorySet::from_texts(
        STDLIB
            .iter()
            .copied()
            .chain([include_str!("../examples/closure2.hex"), named_use]),
    )?;
    let elaborated = catena_lang::elaborate::elaborate(raw)?;
    let theory_set = TheorySet::from_raw(elaborated)?;
    let program = TheoryId(op("program"));
    let selected = BTreeMap::from([(
        program.clone(),
        BTreeSet::from([op("closure2.consume-nested")]),
    )]);
    let inlined = inline_definitions::run(&theory_set, &selected)?;
    let Theory::Theory { arrows, .. } = &inlined.theories[&program] else {
        anyhow::bail!("program should be a user theory");
    };

    anyhow::ensure!(arrows.contains_key(&op("closure2.consume-nested")));
    anyhow::ensure!(arrows.contains_key(&op("name.closure2.consume-nested")));
    let direct = arrows[&op("closure2-nested-direct")]
        .definition
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing direct nested definition"))?;
    anyhow::ensure!(
        direct
            .hypergraph
            .edges
            .iter()
            .all(|operation| operation.as_str() != "closure2.consume-nested")
    );
    let named = arrows[&op("closure2-nested-named-retention")]
        .definition
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing named nested definition"))?;
    anyhow::ensure!(
        named
            .hypergraph
            .edges
            .iter()
            .any(|operation| { operation.as_str() == "name.closure2.consume-nested" })
    );
    Ok(())
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
        .closure_conversion
        .as_ref()
        .map(|conversion| &conversion.closure_forgotten_definitions)
        .and_then(|theories| theories.get(&program))
        .ok_or_else(|| anyhow::anyhow!("forget_closures did not emit the program theory"))?;

    for (definition, expected_regions) in [
        ("closure2-named-if", 2),
        ("closure2-captured-if", 2),
        ("closure2-composed-if", 2),
        ("closure2-tensored-if", 2),
        ("closure2-packed-closure", 0),
        ("closure2-packed-if", 2),
        ("closure2-reduce", 2),
        ("closure2-indexed-if", 2),
        ("closure2-mixed-context-if", 2),
        ("closure2-duplicate-context-if", 2),
        ("closure2-sparse-context-if", 2),
        ("closure2-context-reduce", 2),
        ("closure2-internal-context-if", 2),
        ("closure2-two-sparse-contexts-if", 2),
        ("closure2-nested-direct", 2),
        ("closure2-pair-reduce", 2),
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
            .filter(|(edge, _)| matches!(edge, ClosureForgotten::ClosureMarker))
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
fn closure2_product_examples_generate_renderable_gpu_code() -> anyhow::Result<()> {
    let report = compile_through_closure_conversion(include_str!("../examples/closure2.hex"))?;
    let modules = report
        .gpu_modules
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("closure2 examples did not reach GPU codegen"))?;

    for (definition, operation, input_sizes, output_sizes) in [
        ("closure2-packed-closure", "bool.and", vec![1, 1], vec![1]),
        (
            "closure2-packed-if",
            "bool.ifc",
            vec![1, 1, 1, 1, 1, 0],
            vec![1],
        ),
        (
            "closure2-tensored-if",
            "bool.ifc",
            vec![2, 1, 2, 1, 1, 0],
            vec![2],
        ),
    ] {
        let module = modules
            .values()
            .find(|module| {
                module
                    .source_name
                    .as_ref()
                    .is_some_and(|name| name.as_str() == definition)
            })
            .ok_or_else(|| anyhow::anyhow!("missing GPU module for `{definition}`"))?;
        let assignment = module
            .entry
            .assignments
            .iter()
            .find(|assignment| assignment.op.as_str() == operation)
            .ok_or_else(|| anyhow::anyhow!("`{definition}` does not call `{operation}`"))?;

        anyhow::ensure!(assignment.input_sizes == input_sizes);
        anyhow::ensure!(assignment.output_sizes == output_sizes);
        anyhow::ensure!(assignment.outputs.len() == assignment.output_sizes.iter().sum());
        render_module(module, GpuDialect::Hip)?;
    }

    Ok(())
}

#[test]
fn closure2_matmul_examples_inline_closure_only_helpers() -> anyhow::Result<()> {
    let report = compile_through_closure_conversion(include_str!("../examples/closure2.hex"))?;
    let program = TheoryId(op("program"));
    let Theory::Theory { arrows, .. } = report
        .theory_set
        .as_ref()
        .and_then(|theories| theories.theories.get(&program))
        .ok_or_else(|| anyhow::anyhow!("missing inlined program theory"))?
    else {
        anyhow::bail!("program should be a user theory");
    };

    for inlined in [
        "closure2.matmul-dot",
        "closure2.matmul-cell",
        "closure2.matrix-row",
        "closure2.matrix-col",
        "closure2.f32-buf-view",
        "closure2.row-major-matrix-view",
    ] {
        anyhow::ensure!(
            !arrows.contains_key(&op(inlined)),
            "closure-boundary helper `{inlined}` should have been inlined"
        );
    }

    let forgotten = report
        .closure_conversion
        .as_ref()
        .and_then(|conversion| conversion.closure_forgotten_definitions.get(&program))
        .ok_or_else(|| anyhow::anyhow!("missing forgotten matmul examples"))?;
    let modules = report
        .gpu_modules
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("matmul examples did not reach GPU codegen"))?;

    for outer in [
        "closure2.matmul-two-bufs-at",
        "closure2.matmul-buf-identity-at",
    ] {
        anyhow::ensure!(find_regions(&forgotten[&op(outer)])?.len() == 2);
    }

    for outer in [
        "closure2.matmul-two-bufs",
        "closure2.matmul-buf-and-identity",
    ] {
        let module = modules
            .values()
            .find(|module| {
                module
                    .source_name
                    .as_ref()
                    .is_some_and(|name| name.as_str() == outer)
            })
            .ok_or_else(|| anyhow::anyhow!("missing GPU module for `{outer}`"))?;
        anyhow::ensure!(
            module
                .entry
                .assignments
                .iter()
                .any(|assignment| assignment.op.as_str() == "materializec"),
            "`{outer}` should build its output buffer with materializec"
        );
        for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
            let rendered = render_module(module, dialect)?;
            anyhow::ensure!(rendered.contains("MallocManaged"));
            anyhow::ensure!(rendered.contains("DeviceSynchronize"));
            anyhow::ensure!(rendered.contains("out[i] = value"));
        }
    }

    Ok(())
}

#[test]
fn closure2_finds_named_and_captured_regions() -> anyhow::Result<()> {
    let report = compile_through_closure_conversion(include_str!("../examples/closure2.hex"))?;
    let definitions = report
        .closure_conversion
        .as_ref()
        .map(|conversion| &conversion.closure_forgotten_definitions)
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
        let [edge] = region.edges.as_slice() else {
            anyhow::bail!("captured closure should contain one unit discard");
        };
        anyhow::ensure!(matches!(
            &captured.hypergraph.edges[edge.0],
            ClosureForgotten::Operation(operation) if operation.as_str() == "unit.elim"
        ));
    }

    Ok(())
}

#[test]
fn closure2_builds_closure_and_name_arrows() -> anyhow::Result<()> {
    let report = compile_through_closure_conversion(include_str!("../examples/closure2.hex"))?;
    let generated = report
        .closure_conversion
        .as_ref()
        .map(|conversion| &conversion.generated_theory)
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
    anyhow::ensure!(
        report
            .gpu_modules
            .as_ref()
            .is_some_and(|modules| modules.values().any(|module| {
                module
                    .source_name
                    .as_ref()
                    .is_some_and(|name| name.as_str().starts_with("closure."))
            })),
        "generated closure bodies should reach GPU codegen"
    );
    Ok(())
}

#[test]
fn closure2_replacement_uses_generated_name_boundaries() -> anyhow::Result<()> {
    let report = compile_through_closure_conversion(include_str!("../examples/closure2.hex"))?;
    let theory_id = TheoryId(op("program"));
    let definitions = report
        .closure_conversion
        .as_ref()
        .map(|conversion| &conversion.rewritten_definitions)
        .and_then(|theories| theories.get(&theory_id))
        .ok_or_else(|| anyhow::anyhow!("missing replaced program definitions"))?;
    let Theory::Theory { arrows, .. } = report
        .closure_conversion
        .as_ref()
        .map(|conversion| &conversion.replacement_theory)
        .and_then(|theories| theories.theories.get(&theory_id))
        .ok_or_else(|| anyhow::anyhow!("missing replaced program theory"))?
    else {
        anyhow::bail!("program should be a user theory");
    };
    let final_definitions = report
        .closure_conversion
        .as_ref()
        .map(|conversion| &conversion.runtime_functions)
        .and_then(|theories| theories.get(&theory_id))
        .ok_or_else(|| anyhow::anyhow!("missing final converted program definitions"))?;

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
    anyhow::ensure!(final_definitions.values().all(|term| {
        term.hypergraph
            .edges
            .iter()
            .all(|operation| !operation.as_str().starts_with("__catena_context."))
    }));
    Ok(())
}

#[test]
fn closure2_context_examples_have_expected_pre_codegen_graphs() -> anyhow::Result<()> {
    let report = compile_through_closure_conversion(include_str!("../examples/closure2.hex"))?;
    let program = TheoryId(op("program"));
    let final_definitions = report
        .unpacked_products
        .as_ref()
        .and_then(|theories| theories.get(&program))
        .ok_or_else(|| anyhow::anyhow!("missing final graphs before codegen"))?;
    let Theory::Theory { arrows, .. } = &report
        .closure_conversion
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing closure conversion report"))?
        .generated_theory
        .theories[&program]
    else {
        anyhow::bail!("program should be a user theory");
    };

    assert_all_final_graphs_are_lowered(final_definitions)?;
    assert_product_and_parallel_final_graphs(final_definitions)?;
    assert_context_if_final_graphs(final_definitions, arrows)?;
    assert_context_reduce_final_graph(final_definitions, arrows)?;
    assert_tuple_reduce_final_graph(final_definitions, &report)?;
    assert_distinct_closure_symbols(&report)?;

    Ok(())
}

type FinalTerm = OpenHypergraph<Tree<(), Operation>, OperationWithBoundarySizes<Operation>>;

fn assert_all_final_graphs_are_lowered(
    definitions: &BTreeMap<Operation, FinalTerm>,
) -> anyhow::Result<()> {
    // unpacked_products is the exact graph set handed to codegen. Structural
    // adapters and temporary context projections must be absent everywhere.
    for (definition, term) in definitions {
        assert_final_graph_has_no_conversion_scaffolding(definition.as_str(), term)?;
        if !matches!(
            definition.as_str(),
            "closure2-named-if"
                | "closure2-captured-if"
                | "closure2-composed-if"
                | "closure2-tensored-if"
                | "closure2-packed-closure"
                | "closure2-packed-if"
                | "closure2-nested-direct"
        ) {
            // Contextual programs can deliberately share erased/static wires.
            // Check monogamy on the runtime-only examples where every wire is
            // linear all the way through lowering.
            continue;
        }
        let mut term = term.clone();
        term.quotient().map_err(|error| {
            anyhow::anyhow!("could not quotient final graph `{definition}`: {error:?}")
        })?;
        anyhow::ensure!(
            term.to_strict().is_monogamous(),
            "final graph `{definition}` is not monogamous"
        );
    }
    Ok(())
}

fn assert_product_and_parallel_final_graphs(
    definitions: &BTreeMap<Operation, FinalTerm>,
) -> anyhow::Result<()> {
    // A tensored closure contributes two flattened environment wires. Packing
    // two closures preserves one environment wire for each branch.
    for (definition, expected_sources, expected_targets) in [
        ("closure2-tensored-if", vec![2, 1, 2, 1, 1, 0], vec![2]),
        ("closure2-packed-if", vec![1, 1, 1, 1, 1, 0], vec![1]),
    ] {
        let converted_if = only_final_operation(&definitions[&op(definition)], "bool.ifc")?;
        anyhow::ensure!(converted_if.source_sizes == expected_sources);
        anyhow::ensure!(converted_if.target_sizes == expected_targets);
    }
    let packed_consumer =
        only_final_operation(&definitions[&op("closure2-packed-closure")], "bool.and")?;
    anyhow::ensure!(packed_consumer.source_sizes == vec![1, 1]);
    anyhow::ensure!(packed_consumer.target_sizes == vec![1]);

    // Two adjacent multi-edge regions become one converted primitive with two
    // generated names; neither rewrite may invalidate the other region's ids.
    let composed = &definitions[&op("closure2-composed-if")];
    anyhow::ensure!(final_operation_count(composed, "bool.ifc") == 1);
    anyhow::ensure!(final_operation_count(composed, "bool.if") == 0);
    anyhow::ensure!(
        composed
            .hypergraph
            .edges
            .iter()
            .filter(|edge| {
                edge.operation
                    .as_str()
                    .starts_with("name.closure.closure2-composed-if.")
            })
            .count()
            == 2
    );

    let nested = &definitions[&op("closure2-nested-direct")];
    let nested_if = only_final_operation(nested, "bool.ifc")?;
    anyhow::ensure!(nested_if.source_sizes == vec![1, 1, 1, 1, 1, 0]);
    anyhow::ensure!(nested_if.target_sizes == vec![1]);
    anyhow::ensure!(final_operation_count(nested, "closure2.consume-nested") == 0);
    Ok(())
}

fn assert_context_if_final_graphs(
    definitions: &BTreeMap<Operation, FinalTerm>,
    arrows: &BTreeMap<Operation, metacat::theory::TheoryArrow>,
) -> anyhow::Result<()> {
    for definition in [
        "closure2-indexed-if",
        "closure2-mixed-context-if",
        "closure2-duplicate-context-if",
        "closure2-sparse-context-if",
    ] {
        let term = &definitions[&op(definition)];
        let converted_if = only_final_operation(term, "bool.ifc")?;
        anyhow::ensure!(converted_if.source_sizes == vec![1, 1, 1, 1, 1, 0]);
        anyhow::ensure!(converted_if.target_sizes == vec![1]);
        anyhow::ensure!(final_operation_count(term, "bool.if") == 0);
    }

    // Sparse-context uses original Leaf(2) at the call site even though each
    // generated closure arrow renumbers its local context to Leaf(0).
    for (definition, original_leaf) in [
        ("closure2-indexed-if", 0),
        ("closure2-sparse-context-if", 2),
    ] {
        let term = &definitions[&op(definition)];
        let generated_names = term
            .hypergraph
            .edges
            .iter()
            .zip(&term.hypergraph.adjacency)
            .filter(|(edge, _)| {
                edge.operation
                    .as_str()
                    .starts_with(&format!("name.closure.{definition}."))
            })
            .collect::<Vec<_>>();
        anyhow::ensure!(generated_names.len() == 2);
        for (name, boundary) in generated_names {
            anyhow::ensure!(name.source_sizes == vec![1]);
            let [source] = boundary.sources.as_slice() else {
                anyhow::bail!("`{}` should receive one context wire", name.operation);
            };
            anyhow::ensure!(
                matches!(term.hypergraph.nodes[source.0], Tree::Leaf(leaf, ()) if leaf == original_leaf),
                "`{}` should receive original context Leaf({original_leaf})",
                name.operation
            );
        }
    }

    // Context-selected names are evaluated before these closure boundaries.
    // Their evals remain and the generated closure names are nullary.
    for (definition, expected_names_and_evals) in [
        ("closure2-mixed-context-if", 1),
        ("closure2-duplicate-context-if", 2),
    ] {
        let term = &definitions[&op(definition)];
        anyhow::ensure!(
            final_operation_count(term, "name.closure2-u64-id-for-n") == expected_names_and_evals
        );
        anyhow::ensure!(final_operation_count(term, "eval") == expected_names_and_evals);
        let generated_name_sources = term
            .hypergraph
            .edges
            .iter()
            .filter(|edge| {
                edge.operation
                    .as_str()
                    .starts_with(&format!("name.closure.{definition}."))
            })
            .map(|edge| edge.source_sizes.clone())
            .collect::<Vec<_>>();
        anyhow::ensure!(generated_name_sources == vec![vec![], vec![]]);
    }

    // The public branch is u64 -> u64; n appears only on two internal named
    // operations. The generated name receives one canonical n input.
    let internal = &definitions[&op("closure2-internal-context-if")];
    let internal_if = only_final_operation(internal, "bool.ifc")?;
    anyhow::ensure!(internal_if.source_sizes == vec![2, 1, 0, 1, 1, 1]);
    let internal_name = generated_name_boundaries(internal, "closure2-internal-context-if")
        .into_iter()
        .find(|(name, _)| name.source_sizes == vec![1])
        .ok_or_else(|| anyhow::anyhow!("missing context-dependent internal closure name"))?;
    let [internal_context] = internal_name.1.sources.as_slice() else {
        anyhow::bail!("internal closure name should receive one canonical context wire");
    };
    anyhow::ensure!(matches!(
        internal.hypergraph.nodes[internal_context.0],
        Tree::Leaf(0, ())
    ));
    let internal_arrow = generated_closure_arrow(arrows, "closure2-internal-context-if", 1)?;
    anyhow::ensure!(internal_arrow.type_maps.0.sources == internal_arrow.type_maps.1.sources);

    // The generated closure compacts original leaves [2, 5] to [0, 1], while
    // its final name use remains wired to original Leaf(2), then Leaf(5).
    let sparse = &definitions[&op("closure2-two-sparse-contexts-if")];
    let sparse_if = only_final_operation(sparse, "bool.ifc")?;
    anyhow::ensure!(sparse_if.source_sizes == vec![2, 1, 0, 1, 1, 1]);
    let sparse_name = generated_name_boundaries(sparse, "closure2-two-sparse-contexts-if")
        .into_iter()
        .find(|(name, _)| name.source_sizes == vec![1, 1])
        .ok_or_else(|| anyhow::anyhow!("missing two-context generated name"))?;
    let sparse_context = sparse_name
        .1
        .sources
        .iter()
        .map(|node| sparse.hypergraph.nodes[node.0].clone())
        .collect::<Vec<_>>();
    anyhow::ensure!(sparse_context == vec![Tree::Leaf(2, ()), Tree::Leaf(5, ())]);
    let sparse_arrow = generated_closure_arrow(arrows, "closure2-two-sparse-contexts-if", 2)?;
    anyhow::ensure!(sparse_arrow.type_maps.0.sources == sparse_arrow.type_maps.1.sources);
    Ok(())
}

fn generated_name_boundaries<'a>(
    term: &'a FinalTerm,
    definition: &str,
) -> Vec<(
    &'a OperationWithBoundarySizes<Operation>,
    &'a open_hypergraphs::lax::Hyperedge,
)> {
    term.hypergraph
        .edges
        .iter()
        .zip(&term.hypergraph.adjacency)
        .filter(|(edge, _)| {
            edge.operation
                .as_str()
                .starts_with(&format!("name.closure.{definition}."))
        })
        .collect()
}

fn generated_closure_arrow<'a>(
    arrows: &'a BTreeMap<Operation, metacat::theory::TheoryArrow>,
    definition: &str,
    context_arity: usize,
) -> anyhow::Result<&'a metacat::theory::TheoryArrow> {
    arrows
        .iter()
        .filter(|(name, _)| name.as_str().starts_with(&format!("closure.{definition}.")))
        .map(|(_, arrow)| arrow)
        .find(|arrow| arrow.type_maps.0.sources.len() == context_arity)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "missing generated closure for `{definition}` with context arity {context_arity}"
            )
        })
}

fn assert_context_reduce_final_graph(
    definitions: &BTreeMap<Operation, FinalTerm>,
    arrows: &BTreeMap<Operation, metacat::theory::TheoryArrow>,
) -> anyhow::Result<()> {
    // reducec receives accumulator environment/name, producer environment/name,
    // and length. Only the indexed producer name depends on n.
    let reduce = &definitions[&op("closure2-context-reduce")];
    let converted_reduce = only_final_operation(reduce, "reducec")?;
    anyhow::ensure!(converted_reduce.source_sizes == vec![1, 0, 1, 1, 1, 1]);
    anyhow::ensure!(converted_reduce.target_sizes == vec![1]);
    anyhow::ensure!(final_operation_count(reduce, "reduce") == 0);
    let mut generated_name_sources = reduce
        .hypergraph
        .edges
        .iter()
        .filter(|edge| {
            edge.operation
                .as_str()
                .starts_with("name.closure.closure2-context-reduce.")
        })
        .map(|edge| edge.source_sizes.clone())
        .collect::<Vec<_>>();
    generated_name_sources.sort();
    anyhow::ensure!(generated_name_sources == vec![vec![], vec![1]]);

    // The indexed producer has n on both type-map domains even though only its
    // source object mentions ix n and its target is u64.
    let indexed_producer = arrows
        .iter()
        .filter(|(name, _)| {
            name.as_str()
                .starts_with("closure.closure2-context-reduce.")
        })
        .map(|(_, arrow)| arrow)
        .find(|arrow| arrow.raw.type_maps.0.to_string().contains("ix"))
        .ok_or_else(|| anyhow::anyhow!("missing generated indexed producer"))?;
    anyhow::ensure!(indexed_producer.type_maps.0.sources == indexed_producer.type_maps.1.sources);
    anyhow::ensure!(indexed_producer.type_maps.0.sources.len() == 1);
    Ok(())
}

fn assert_tuple_reduce_final_graph(
    definitions: &BTreeMap<Operation, FinalTerm>,
    report: &CompileReport,
) -> anyhow::Result<()> {
    let term = &definitions[&op("closure2-pair-reduce")];
    let reduce = only_final_operation(term, "reducec")?;
    anyhow::ensure!(reduce.source_sizes == vec![2, 0, 1, 1, 1, 1]);
    anyhow::ensure!(reduce.target_sizes == vec![2]);
    anyhow::ensure!(final_operation_count(term, "reduce") == 0);

    let module = report
        .gpu_modules
        .as_ref()
        .and_then(|modules| {
            modules.values().find(|module| {
                module
                    .source_name
                    .as_ref()
                    .is_some_and(|name| name.as_str() == "closure2-pair-reduce")
            })
        })
        .ok_or_else(|| anyhow::anyhow!("missing tuple-reduce GPU module"))?;
    for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
        render_module(module, dialect)?;
    }
    Ok(())
}

fn assert_distinct_closure_symbols(report: &CompileReport) -> anyhow::Result<()> {
    let modules = report
        .gpu_modules
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing GPU modules"))?;
    let named_if = modules
        .values()
        .find(|module| {
            module
                .source_name
                .as_ref()
                .is_some_and(|name| name.as_str() == "closure2-named-if")
        })
        .ok_or_else(|| anyhow::anyhow!("missing named-if GPU module"))?;
    let converted_if = named_if
        .entry
        .assignments
        .iter()
        .find(|assignment| assignment.op.as_str() == "bool.ifc")
        .ok_or_else(|| anyhow::anyhow!("named-if module has no bool.ifc"))?;
    let symbols = converted_if
        .inputs
        .iter()
        .filter_map(|input| match input {
            GpuValue::FnSymbol(symbol) => Some(symbol.target.clone()),
            GpuValue::Var(_) => None,
        })
        .collect::<BTreeSet<_>>();
    anyhow::ensure!(
        symbols.len() == 2,
        "named branches must use distinct symbols"
    );
    for symbol in symbols {
        anyhow::ensure!(modules.values().any(|module| {
            module
                .source_name
                .as_ref()
                .is_some_and(|source| source == &symbol)
        }));
    }
    Ok(())
}

fn only_final_operation<'a>(
    term: &'a FinalTerm,
    operation: &str,
) -> anyhow::Result<&'a OperationWithBoundarySizes<Operation>> {
    let matches = term
        .hypergraph
        .edges
        .iter()
        .filter(|edge| edge.operation.as_str() == operation)
        .collect::<Vec<_>>();
    let [edge] = matches.as_slice() else {
        anyhow::bail!("expected one `{operation}` edge, found {}", matches.len());
    };
    Ok(edge)
}

fn final_operation_count(term: &FinalTerm, operation: &str) -> usize {
    term.hypergraph
        .edges
        .iter()
        .filter(|edge| edge.operation.as_str() == operation)
        .count()
}

fn assert_final_graph_has_no_conversion_scaffolding(
    definition: &str,
    term: &FinalTerm,
) -> anyhow::Result<()> {
    for operation in &term.hypergraph.edges {
        anyhow::ensure!(
            !operation
                .operation
                .as_str()
                .starts_with("__catena_context."),
            "`{definition}` retained context scaffolding before codegen"
        );
        anyhow::ensure!(
            !matches!(
                operation.operation.as_str(),
                "*.intro" | "*.elim" | "unit.intro" | "unit.elim"
            ),
            "`{definition}` retained structural operation `{}` before codegen",
            operation.operation
        );
    }
    Ok(())
}

fn op(name: &str) -> Operation {
    name.parse().expect("test operation should parse")
}
