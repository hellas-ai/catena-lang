use super::super::compile_through_closure_conversion_with_sources;

const DIRECT_NAME_CONTEXT: &str = include_str!("fixtures/direct_name_context.hex");
const COPIED_NAME_CONTEXT: &str = include_str!("fixtures/copied_name_context.hex");
const ELIM_NAME_CONTEXT: &str = include_str!("fixtures/elim_name_context.hex");
const NAME_CONTEXT_HELPERS: &str = include_str!("fixtures/name_context_helpers.hex");
const NESTED_CONTEXT_VIEW: &str = include_str!("fixtures/nested_context_view.hex");

// Expected-success end-to-end regressions for static context dependencies of
// generated closures.
#[test]
fn direct_metavariable_context_reaches_named_closure() {
    compile_through_closure_conversion_with_sources([NAME_CONTEXT_HELPERS, DIRECT_NAME_CONTEXT])
        .expect("a direct metavariable dependency of name.f should compile");
}

#[test]
fn copied_metavariable_context_reaches_both_named_closures() {
    compile_through_closure_conversion_with_sources([NAME_CONTEXT_HELPERS, COPIED_NAME_CONTEXT])
        .expect("distinct metavariable copies should reach their named closures");
}

#[test]
fn product_elimination_feeding_name_belongs_to_closure_region() {
    compile_through_closure_conversion_with_sources([NAME_CONTEXT_HELPERS, ELIM_NAME_CONTEXT])
        .expect("the parameter path from *.elim to name.f should belong to the closure region");
}

#[test]
fn nested_context_view_belongs_to_outer_closure() {
    compile_through_closure_conversion_with_sources([NAME_CONTEXT_HELPERS, NESTED_CONTEXT_VIEW])
        .expect("a nested context-dependent view should belong to the outer closure region");
}
