use std::{collections::BTreeMap, sync::OnceLock};

use catena_lang::{
    closure::{Conversion, region::ClosureRegion},
    pass::record_boundary_sizes::OperationWithBoundarySizes,
    report::CompileReport,
};
use hexpr::Operation;
use metacat::{
    theory::{Theory, TheoryArrow, TheoryId},
    tree::Tree,
};
use open_hypergraphs::lax::OpenHypergraph;

pub type FinalTerm = OpenHypergraph<Tree<(), Operation>, OperationWithBoundarySizes<Operation>>;

/// Compile the closure fixture once and share its immutable pass report among
/// the integration tests.  Calling `compile` exercises elaboration, checking,
/// closure-boundary inlining, forgetting, conversion, and subsequent lowering.
pub fn report() -> &'static CompileReport {
    static REPORT: OnceLock<CompileReport> = OnceLock::new();
    REPORT.get_or_init(|| {
        let sources = [
            include_str!("closures/fixtures/basic.hex"),
            include_str!("closures/fixtures/products.hex"),
            include_str!("closures/fixtures/context.hex"),
            include_str!("closures/fixtures/reduce.hex"),
            include_str!("closures/fixtures/matmul.hex"),
        ];
        let report =
            crate::compile_with_sources(sources).expect("closure compile fixtures should compile");
        assert!(report.closure_conversion.is_some());
        assert!(report.unpacked_products.is_some());
        report
    })
}

pub fn conversion() -> &'static Conversion {
    report()
        .closure_conversion
        .as_ref()
        .expect("compile should record closure conversion")
}

pub fn program() -> TheoryId {
    TheoryId(op("program"))
}

pub fn regions(definition: &str) -> &'static [ClosureRegion] {
    &conversion().regions[&program()][&op(definition)]
}

pub fn final_term(definition: &str) -> &'static FinalTerm {
    &report().unpacked_products.as_ref().expect("checked above")[&program()][&op(definition)]
}

pub fn generated_arrows() -> &'static BTreeMap<Operation, TheoryArrow> {
    let Theory::Theory { arrows, .. } = &conversion().generated_theory.theories[&program()] else {
        panic!("program should be a user theory");
    };
    arrows
}

pub fn generated_with_prefix(prefix: &str) -> Vec<(&'static Operation, &'static TheoryArrow)> {
    generated_arrows()
        .iter()
        .filter(|(name, _)| name.as_str().starts_with(prefix))
        .collect()
}

pub fn operation_count(term: &FinalTerm, operation: &str) -> usize {
    term.hypergraph
        .edges
        .iter()
        .filter(|edge| edge.operation.as_str() == operation)
        .count()
}

pub fn only_operation<'a>(
    term: &'a FinalTerm,
    operation: &str,
) -> &'a OperationWithBoundarySizes<Operation> {
    let matches = term
        .hypergraph
        .edges
        .iter()
        .filter(|edge| edge.operation.as_str() == operation)
        .collect::<Vec<_>>();
    let [edge] = matches.as_slice() else {
        panic!("expected one `{operation}` edge, found {}", matches.len());
    };
    edge
}

pub fn assert_fully_lowered(definition: &str) {
    for edge in &final_term(definition).hypergraph.edges {
        let operation = edge.operation.as_str();
        assert!(
            !operation.starts_with("__catena_context."),
            "`{definition}` retained context scaffolding"
        );
        assert!(
            !matches!(operation, "*.intro" | "*.elim" | "unit.intro" | "unit.elim"),
            "`{definition}` retained structural operation `{operation}`"
        );
    }
}

/// Runtime-only definitions should remain linear after all graph rewrites and
/// product lowering. Contextual definitions are checked separately because
/// erased/type-level context wires may intentionally be shared.
pub fn assert_monogamous(definition: &str) {
    let mut term = final_term(definition).clone();
    term.quotient()
        .unwrap_or_else(|error| panic!("could not quotient `{definition}`: {error:?}"));
    assert!(
        term.to_strict().is_monogamous(),
        "final graph `{definition}` is not monogamous"
    );
}

pub fn op(name: &str) -> Operation {
    name.parse().expect("test operation should parse")
}
