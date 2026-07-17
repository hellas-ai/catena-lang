use crate::support::*;

/// Source-level `materialize` consumes an indexed closure.
/// Closure conversion should replace it with the explicit `materializec` ABI:
///
/// ```text
/// len, producer-closure ─▶ materialize
/// len, env, producer-name ─▶ materializec
/// ```
#[test]
fn source_materialize_becomes_explicit_function_pair() {
    assert_eq!(regions("materialize-indexes-source").len(), 1);
    let term = final_term("materialize-indexes-source");
    assert_eq!(operation_count(term, "materialize"), 0);
    assert_eq!(operation_count(term, "materializec"), 1);

    let materialize = only_operation(term, "materializec");
    assert_eq!(materialize.source_sizes, vec![1, 1, 1]);
    assert_eq!(materialize.target_sizes, vec![1]);
    assert_fully_lowered("materialize-indexes-source");
}
