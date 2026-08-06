use super::super::compile_with_sources;

const CLOSURE_CAPTURE_ORDER: &str = include_str!("fixtures/closure_capture_order.hex");

#[test]
fn closure_producers_are_converted_before_the_capturing_region() {
    compile_with_sources([CLOSURE_CAPTURE_ORDER])
        .expect("closure producers should be converted before the capturing region");
}
