use hexpr::Operation;
use metacat::{
    theory::{RawTheorySet, Theory, TheoryId},
    tree::Tree,
};

use crate::{
    closure2::{Conversion, region::find_regions},
    compile::compile,
    pass::forget_closures::ClosureForgotten,
    stdlib,
};

fn conversion(source: &'static str) -> Conversion {
    let raw = RawTheorySet::from_texts(stdlib::sources().chain([source]))
        .expect("ported closure test should parse");
    compile(raw)
        .expect("ported closure test should compile")
        .closure_conversion
        .expect("compile should run closure2")
}

fn program_arrows(
    conversion: &Conversion,
) -> &std::collections::BTreeMap<Operation, metacat::theory::TheoryArrow> {
    let Theory::Theory { arrows, .. } = conversion
        .generated_theory
        .theories
        .get(&TheoryId(op("program")))
        .expect("program theory should exist")
    else {
        panic!("program should be a user theory");
    };
    arrows
}

fn operation_count(
    term: &open_hypergraphs::lax::OpenHypergraph<Tree<(), Operation>, Operation>,
    operation: &str,
) -> usize {
    term.hypergraph
        .edges
        .iter()
        .filter(|candidate| candidate.as_str() == operation)
        .count()
}

#[test]
fn identity_has_no_closure_work_to_do() {
    let conversion = conversion(
        r#"
        (def program closure2-test-id : (bool val) -> (bool val) = [x])
        "#,
    );
    let program = TheoryId(op("program"));
    let forgotten = &conversion.closure_forgotten_definitions[&program][&op("closure2-test-id")];

    assert!(find_regions(forgotten).unwrap().is_empty());
    assert!(
        forgotten
            .hypergraph
            .edges
            .iter()
            .all(|edge| !matches!(edge, ClosureForgotten::ClosureMarker))
    );
    assert!(
        program_arrows(&conversion)
            .keys()
            .all(|name| !name.as_str().starts_with("closure.closure2-test-id."))
    );
}

#[test]
fn named_if_generates_two_closures_and_ifc() {
    let conversion = conversion(
        r#"
        (def program closure2-test-named-if :
          {(bool val) (bool val)} -> (bool val)
        = ([flag argument.]
          {(name.bool.not lift) (name.bool.id lift) [.flag argument]}
          bool.if
        ))
        "#,
    );
    let program = TheoryId(op("program"));
    let forgotten =
        &conversion.closure_forgotten_definitions[&program][&op("closure2-test-named-if")];
    let regions = find_regions(forgotten).unwrap();

    assert_eq!(regions.len(), 2);
    assert!(regions.iter().all(|region| region.environment.is_empty()));

    let arrows = program_arrows(&conversion);
    assert_eq!(
        arrows
            .keys()
            .filter(|name| name.as_str().starts_with("closure.closure2-test-named-if."))
            .count(),
        2
    );
    assert_eq!(
        arrows
            .keys()
            .filter(|name| name
                .as_str()
                .starts_with("name.closure.closure2-test-named-if."))
            .count(),
        2
    );

    let rewritten = &conversion.rewritten_definitions[&program][&op("closure2-test-named-if")];
    assert_eq!(operation_count(rewritten, "bool.ifc"), 1);
    assert_eq!(operation_count(rewritten, "bool.if"), 0);
}

#[test]
fn deferred_values_become_runtime_environments() {
    let conversion = conversion(
        r#"
        (def program closure2-test-captured-if :
          {(bool val) (bool val) (bool val)} -> (bool val)
        = ([lhs rhs flag.]
          {([.lhs] defer) ([.rhs] defer) [.flag] unit.intro}
          bool.if
        ))
        "#,
    );
    let program = TheoryId(op("program"));
    let forgotten =
        &conversion.closure_forgotten_definitions[&program][&op("closure2-test-captured-if")];
    let regions = find_regions(forgotten).unwrap();

    assert_eq!(regions.len(), 2);
    assert!(regions.iter().all(|region| region.environment.len() == 1));
    assert!(regions.iter().all(|region| region.edges.is_empty()));

    let rewritten = &conversion.rewritten_definitions[&program][&op("closure2-test-captured-if")];
    assert_eq!(operation_count(rewritten, "bool.ifc"), 1);
}

#[test]
fn product_capture_stays_one_packed_environment() {
    let conversion = conversion(
        r#"
        (def program closure2-test-product-capture :
          {({(bool val) (bool val)} *) (bool val)}
          ->
          ({(bool val) (bool val)} *)
        = ([captured flag.]
          {
            ([.captured] defer)
            ({bool.f bool.t} *.intro defer)
            [.flag]
            unit.intro
          }
          bool.if
        ))
        "#,
    );
    let program = TheoryId(op("program"));
    let generated = conversion
        .generated_functions
        .get(&program)
        .expect("captured closures should generate functions");
    let captured = generated
        .iter()
        .find(|(_, term)| {
            term.sources.first().is_some_and(|source| {
                matches!(
                    &term.hypergraph.nodes[source.0],
                    Tree::Node(operation, _, _) if operation.as_str() == "*"
                )
            })
        })
        .map(|(_, term)| term)
        .expect("one generated closure should carry the captured product");

    assert_eq!(
        captured.sources.len(),
        2,
        "generated body should receive environment and unit domain"
    );
    assert_eq!(captured.targets.len(), 1);
}

#[test]
fn indexed_defer_supplies_ambient_context() {
    let conversion = conversion(
        r#"
        (def program closure2-test-indexed-if :
          ([n.] {([.n] ix val) ([.n] ix val) (bool val)})
          ->
          ([n.] ([.n] ix val))
        = ([i j flag.]
          {([.i] defer) ([.j] defer) [.flag] unit.intro}
          bool.if
        ))
        "#,
    );
    let arrows = program_arrows(&conversion);
    let generated_names = arrows
        .iter()
        .filter(|(name, _)| {
            name.as_str()
                .starts_with("name.closure.closure2-test-indexed-if.")
        })
        .collect::<Vec<_>>();

    assert_eq!(generated_names.len(), 2);
    assert!(
        generated_names
            .iter()
            .all(|(_, arrow)| arrow.type_maps.0.targets.len() == 1)
    );
}

#[test]
fn reduce_converts_both_closure_arguments() {
    let conversion = conversion(
        r#"
        (def program closure2-test-one-at :
          ([n.] ([.n] ix val)) -> ([n.] (u64 val))
        = ([i.] u64.one))

        (def program closure2-test-reduce : [] -> (u64 val) = (
          {
            const.u64.0x0000000000000000
            (name.u64.add lift)
            ((u64.zero :.param) name.closure2-test-one-at lift)
            u64.zero
          }
          reduce
        ))
        "#,
    );
    let program = TheoryId(op("program"));
    let forgotten =
        &conversion.closure_forgotten_definitions[&program][&op("closure2-test-reduce")];
    assert_eq!(find_regions(forgotten).unwrap().len(), 2);

    let rewritten = &conversion.rewritten_definitions[&program][&op("closure2-test-reduce")];
    assert_eq!(operation_count(rewritten, "reducec"), 1);
    assert_eq!(operation_count(rewritten, "reduce"), 0);
}

#[test]
fn mixed_runtime_and_free_context_name_boundary() {
    let conversion = conversion(
        r#"
        (def program closure2-test-u64-id-for-n :
          ([n.] (u64 val)) -> ([n.] (u64 val))
        = ([x.] [.x]))

        (def program closure2-test-mixed-context :
          ([n.] {({[.n] u64} :) (u64 val) (bool val)})
          ->
          ([n.] (u64 val))
        = ([len x flag.]
          ([.len] :.ty [runtime-len closure-n.]
            {
              ({([.x] defer) ([.closure-n] name.closure2-test-u64-id-for-n lift)} compose)
              ([.runtime-len] :.forget defer)
              [.flag]
              unit.intro
            }
            bool.if
          )
        ))
        "#,
    );
    let program = TheoryId(op("program"));
    let rewritten = &conversion.rewritten_definitions[&program][&op("closure2-test-mixed-context")];

    assert_eq!(operation_count(rewritten, "bool.ifc"), 1);
    assert!(rewritten.hypergraph.edges.iter().any(|operation| {
        operation
            .as_str()
            .starts_with("name.closure.closure2-test-mixed-context.")
    }));
}

#[test]
fn parallel_composed_regions_rewrite_without_stale_ids() {
    let conversion = conversion(
        r#"
        (def program closure2-test-parallel-regions :
          {(bool val) (bool val) (bool val)} -> (bool val)
        = ([lhs rhs flag.]
          {
            ({([.lhs] defer) (name.bool.not lift)} compose)
            ({([.rhs] defer) (name.bool.id lift)} compose)
            [.flag]
            unit.intro
          }
          bool.if
        ))
        "#,
    );
    let program = TheoryId(op("program"));
    let forgotten =
        &conversion.closure_forgotten_definitions[&program][&op("closure2-test-parallel-regions")];
    assert_eq!(find_regions(forgotten).unwrap().len(), 2);

    let rewritten =
        &conversion.rewritten_definitions[&program][&op("closure2-test-parallel-regions")];
    assert_eq!(operation_count(rewritten, "bool.ifc"), 1);
    assert!(rewritten.hypergraph.edges.iter().all(|operation| {
        !operation.as_str().starts_with("__catena_context.closure.")
            || operation
                .as_str()
                .contains("closure2-test-parallel-regions")
    }));
}

#[test]
fn context_dependent_reduce_closures_share_ambient_length() {
    let conversion = conversion(
        r#"
        (def program closure2-test-diagonal-view :
          ([n.] ([.n] ix val)) -> ([n.] (u64 val))
        = ([i.] u64.one))

        (def program closure2-test-reduce-diagonal :
          ([n.] {({[.n] u64} :) ({[.n] u64} :)})
          ->
          ([n.] (u64 val))
        = ([producer-len reduce-len.]
          {
            const.u64.0x0000000000000000
            (name.u64.add lift)
            (([.producer-len] :.param) name.closure2-test-diagonal-view lift)
            [.reduce-len]
          }
          reduce
        ))
        "#,
    );
    let program = TheoryId(op("program"));
    let rewritten =
        &conversion.rewritten_definitions[&program][&op("closure2-test-reduce-diagonal")];

    assert_eq!(operation_count(rewritten, "reducec"), 1);
    let arrows = program_arrows(&conversion);
    let names = arrows
        .iter()
        .filter(|(name, _)| {
            name.as_str()
                .starts_with("name.closure.closure2-test-reduce-diagonal.")
        })
        .collect::<Vec<_>>();
    assert_eq!(names.len(), 2);
    let mut context_arities = names
        .iter()
        .map(|(_, arrow)| arrow.type_maps.0.targets.len())
        .collect::<Vec<_>>();
    context_arities.sort_unstable();
    assert_eq!(
        context_arities,
        vec![0, 1],
        "closed accumulator should be nullary; only the indexed producer needs ambient n"
    );
}

fn op(name: &str) -> Operation {
    name.parse().expect("test operation should parse")
}
