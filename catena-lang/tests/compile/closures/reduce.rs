use crate::support::*;

/// `reduce` consumes two closures with different domains. Both must be found
/// and replaced together by one `reducec` operation.
///
/// ```text
/// init, accumulator-closure, producer-closure, n ─▶ reduce
/// init, acc-env, acc-name, prod-env, prod-name, n ─▶ reducec
/// ```
#[test]
fn scalar_reduce_converts_both_closure_arguments() {
    assert_eq!(regions("scalar-reduce").len(), 2);
    let term = final_term("scalar-reduce");
    assert_eq!(operation_count(term, "reduce"), 0);
    assert_eq!(operation_count(term, "reducec"), 1);
    assert_fully_lowered("scalar-reduce");
}

/// Minimal context is computed independently for each generated closure: the
/// closed accumulator name is nullary, while the indexed producer needs `n`.
///
/// ```text
/// name.accumulator : () ─▶ fn-ptr
/// name.producer    :  n ─▶ fn-ptr
/// ```
#[test]
fn reduce_names_use_only_the_context_their_bodies_need() {
    let term = final_term("context-reduce");
    let reduce = only_operation(term, "reducec");
    assert_eq!(reduce.source_sizes, vec![1, 0, 1, 1, 1, 1]);
    assert_eq!(reduce.target_sizes, vec![1]);

    let mut name_sources = term
        .hypergraph
        .edges
        .iter()
        .filter(|edge| {
            edge.operation
                .as_str()
                .starts_with("name.closure.context-reduce.")
        })
        .map(|edge| edge.source_sizes.clone())
        .collect::<Vec<_>>();
    name_sources.sort();
    assert_eq!(name_sources, vec![vec![], vec![1]]);
    assert_fully_lowered("context-reduce");
}

/// Product accumulators are flattened consistently on both the environment and
/// result sides of the converted primitive.
#[test]
fn product_accumulator_has_matching_expanded_boundaries() {
    let reduce = only_operation(final_term("pair-reduce"), "reducec");
    assert_eq!(reduce.source_sizes, vec![2, 0, 1, 1, 1, 1]);
    assert_eq!(reduce.target_sizes, vec![2]);
    assert_fully_lowered("pair-reduce");
}
