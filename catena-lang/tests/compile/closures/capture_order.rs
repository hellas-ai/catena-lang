use super::super::compile_with_sources;

const CLOSURE_CAPTURE_ORDER: &str = include_str!("fixtures/closure_capture_order.hex");

#[test]
fn captured_closures_are_converted_before_their_outer_closure() {
    compile_with_sources([CLOSURE_CAPTURE_ORDER])
        .expect("captured closures should be converted before their outer closure");
}
