use crate::support::*;

/// A closure with a product-valued environment is flattened only at the final
/// boundary: `bool.ifc` receives two runtime environment wires per branch.
#[test]
fn product_environments_are_expanded_at_primitive_boundaries() {
    assert_eq!(regions("tensored-if").len(), 2);
    let converted_if = only_operation(final_term("tensored-if"), "bool.ifc");
    assert_eq!(converted_if.source_sizes, vec![2, 1, 2, 1, 1, 0]);
    assert_eq!(converted_if.target_sizes, vec![2]);
    assert_fully_lowered("tensored-if");
}

/// Packing and immediately unpacking closures must not change their arity or
/// require special replacement logic. The structural pass removes the product
/// round trip after the two regions have been converted.
///
/// ```text
/// closure a ─┐   *.intro   *.elim ┌─ closure a
/// closure b ─┴─▶ (a * b) ────────┴─ closure b ─▶ bool.if
///
/// final:  env-a, name-a, env-b, name-b, flag, unit ─▶ bool.ifc
/// ```
#[test]
fn packed_closures_cross_structural_boundaries() {
    assert_eq!(regions("packed-if").len(), 2);
    let converted_if = only_operation(final_term("packed-if"), "bool.ifc");
    assert_eq!(converted_if.source_sizes, vec![1, 1, 1, 1, 1, 0]);
    assert_eq!(converted_if.target_sizes, vec![1]);

    let packed_consumer = only_operation(final_term("packed-closure"), "bool.and");
    assert_eq!(packed_consumer.source_sizes, vec![1, 1]);
    assert_eq!(packed_consumer.target_sizes, vec![1]);
    assert_fully_lowered("packed-if");
    assert_fully_lowered("packed-closure");
}

/// Definitions whose public boundary contains closures are expanded before
/// forgetting. Their calls disappear, while the surrounding program still
/// reaches conversion and product lowering normally.
///
/// ```text
/// nested-direct ──calls──▶ consume-nested
///       inline closure boundary
/// nested-direct ─────────▶ expanded consumer body ─▶ conversion
/// ```
#[test]
fn closure_boundary_helpers_are_inlined_before_forgetting() {
    let final_graph = final_term("nested-direct");
    assert_eq!(operation_count(final_graph, "consume-nested"), 0);
    assert_eq!(operation_count(final_graph, "bool.ifc"), 1);
    let converted_if = only_operation(final_graph, "bool.ifc");
    assert_eq!(converted_if.source_sizes, vec![1, 1, 1, 1, 1, 0]);
    assert_fully_lowered("nested-direct");
}

/// A named function may consume a packed ordinary value and capture another
/// value while it is composed as a closure. Forgetting must preserve the named
/// call's product boundary and erase only the surrounding CMC plumbing.
#[test]
fn named_function_keeps_its_packed_argument_boundary() {
    let term = final_term("run-named-and-packed-with-free");
    assert_eq!(operation_count(term, "name.and-packed-with-free"), 1);
    assert_eq!(operation_count(term, "eval"), 1);
    assert_eq!(operation_count(term, "compose"), 0);
    assert_eq!(operation_count(term, "run"), 0);
    assert_fully_lowered("run-named-and-packed-with-free");
}

/// Product packing, inlining, and closure replacement must not introduce an
/// implicit fan-out. These context-free cases can therefore be checked as
/// strict monogamous graphs after product lowering.
#[test]
fn product_final_graphs_are_monogamous() {
    for definition in [
        "tensored-if",
        "packed-closure",
        "packed-if",
        "nested-direct",
        "run-named-and-packed-with-free",
    ] {
        assert_monogamous(definition);
    }
}
