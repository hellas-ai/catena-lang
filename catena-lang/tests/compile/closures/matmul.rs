use metacat::theory::Theory;

use crate::support::*;

/// The application-sized fixture builds matrix-view closures outside the
/// materialization producer. It composes a partially applied row-major index
/// decoder with a partially applied `cell-dot`; `cell-dot` in turn partially
/// applies `product-at` to obtain its function of `k`. After forgetting, those
/// named evaluations are specialized by inlining their forgotten bodies;
/// nested reduction closures are converted inside-out before the outer
/// materialization producer.
///
/// ```text
/// buf + buf ─▶ two closures ─┐
///                            ├─▶ indices ; partial cell-dot ─▶ materializec
/// buf + id  ─▶ two closures ─┘
/// ```
#[test]
fn matmul_entry_points_share_inlined_closure_only_logic() {
    let Theory::Theory { arrows, .. } = &report()
        .theory_set
        .as_ref()
        .expect("compile checked above")
        .theories[&program()]
    else {
        panic!("program should be a user theory");
    };
    for helper in [
        "f32.row-major.matrix-view",
        "f32.matmul.row-major.materialize",
    ] {
        assert!(
            !arrows.contains_key(&op(helper)),
            "`{helper}` should be inlined"
        );
    }

    assert!(arrows.contains_key(&op("f32.matmul.cell-dot")));
    assert!(arrows.contains_key(&op("name.f32.matmul.cell-dot")));
    assert!(arrows.contains_key(&op("f32.matmul.product-at")));
    assert!(arrows.contains_key(&op("name.f32.matmul.product-at")));
    assert!(!arrows.contains_key(&op("f32.matmul.row-major.cell-at")));
    assert!(!arrows.contains_key(&op("name.f32.matmul.row-major.cell-at")));
    assert!(!arrows.contains_key(&op("matmul-two-bufs-at")));
    assert!(!arrows.contains_key(&op("matmul-buf-identity-at")));

    for entry_point in ["matmul-two-bufs", "matmul-buf-and-identity"] {
        assert!(regions(entry_point).len() >= 3);
        assert_eq!(operation_count(final_term(entry_point), "materializec"), 1);
        assert_eq!(
            operation_count(final_term(entry_point), "f32.matmul.cell-dot"),
            0
        );
        assert_fully_lowered(entry_point);
    }
}
