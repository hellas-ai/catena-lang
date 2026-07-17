use metacat::theory::Theory;

use crate::support::*;

/// The application-sized fixture keeps only its concrete buffer adapters and
/// uses the matrix stdlib for buffer views, row-major views, identity, and
/// closure-only matmul. Compile must inline every stdlib helper with a closure
/// boundary before forgetting and discover both closure arguments at each
/// adapter call.
///
/// ```text
/// buf + buf ─▶ two closures ─┐
///                            ├─▶ stdlib matmul ─▶ materialize ─▶ materializec
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
        "f32.matmul.cell-dot",
        "f32.matmul.row-major.cell-at",
        "f32.matrix.row-view",
        "f32.matrix.col-view",
        "f32.buf.view",
        "f32.row-major.matrix-view",
    ] {
        assert!(
            !arrows.contains_key(&op(helper)),
            "`{helper}` should be inlined"
        );
    }

    for adapter in ["matmul-two-bufs-at", "matmul-buf-identity-at"] {
        assert_eq!(regions(adapter).len(), 2);
        assert_fully_lowered(adapter);
    }

    for entry_point in ["matmul-two-bufs", "matmul-buf-and-identity"] {
        assert_eq!(operation_count(final_term(entry_point), "materializec"), 1);
        assert_fully_lowered(entry_point);
    }
}
