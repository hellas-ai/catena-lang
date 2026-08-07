use metacat::theory::TheoryId;

use super::super::{compile_with_sources, support::op};

const DISCARD_TERMINATED: &str = include_str!("fixtures/discard_terminated.hex");

#[test]
fn discarded_computation_remains_in_converted_closure() {
    // Use the public compiler entry point so elaboration, inlining, closure
    // forgetting, region discovery, conversion, and later lowering all run.
    let report = compile_with_sources([DISCARD_TERMINATED])
        .expect("a discard-terminated closure region should compile end to end");

    let final_terms = report
        .unpacked_products
        .expect("the complete pipeline should reach product lowering");
    let definitions = &final_terms[&TheoryId(op("program"))];
    let (_, closure_body) = definitions
        .iter()
        .find(|(name, _)| {
            name.as_str()
                .starts_with("closure.discard-terminated-region.")
        })
        .expect("closure conversion should generate the discarding branch body");

    // The result of bool.not is unused, but bool.not itself must remain inside
    // the converted closure because discarded computations may have effects.
    assert_eq!(
        closure_body
            .hypergraph
            .edges
            .iter()
            .filter(|edge| edge.operation.as_str() == "bool.not")
            .count(),
        1
    );
}
