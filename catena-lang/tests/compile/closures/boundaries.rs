use super::super::compile_with_sources;

const PRODUCT_DOMAIN: &str = include_str!("fixtures/product_domain.hex");
const PRODUCT_RESULT: &str = include_str!("fixtures/product_result.hex");
const PRODUCT_DOMAIN_AND_RESULT: &str = include_str!("fixtures/product_domain_and_result.hex");

#[test]
fn product_closure_domain_compiles_through_codegen() {
    compile_with_sources([PRODUCT_DOMAIN])
        .expect("a product closure domain should compile through code generation");
}

#[test]
fn product_closure_result_compiles_through_codegen() {
    compile_with_sources([PRODUCT_RESULT])
        .expect("a product closure result should compile through code generation");
}

#[test]
fn product_closure_domain_and_result_compile_through_codegen() {
    compile_with_sources([PRODUCT_DOMAIN_AND_RESULT])
        .expect("product closure domains and results should compile through code generation");
}
