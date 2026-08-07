use super::super::compile_through_closure_conversion_with_sources;

const POST_EVAL_PRODUCT_ESCAPE: &str = include_str!("fixtures/post_eval_product_escape.hex");
const MANUAL_POST_EVAL_PRODUCT_ESCAPE: &str =
    include_str!("fixtures/manual_post_eval_product_escape.hex");

#[test]
fn post_eval_product_split_does_not_split_region_ownership() {
    compile_through_closure_conversion_with_sources([POST_EVAL_PRODUCT_ESCAPE])
        .expect("a post-eval product split should remain in one closure region");
}

#[test]
fn manual_post_eval_product_split_does_not_split_region_ownership() {
    compile_through_closure_conversion_with_sources([MANUAL_POST_EVAL_PRODUCT_ESCAPE])
        .expect("a manually constructed post-eval product split should remain in one region");
}
