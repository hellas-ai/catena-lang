use super::*;

const IMPLICIT_2D_FLATTEN: &str = include_str!("indexing/implicit_2d_flatten.hex");

#[test]
fn two_dimensional_indices_require_an_explicit_layout() {
    let raw = RawTheorySet::from_texts(stdlib::sources().chain([IMPLICIT_2D_FLATTEN])).unwrap();
    let failure = compile(raw).expect_err("a 2D index must not implicitly become a scalar");

    let CompileError::Check(CheckError::Definition { definition, .. }) = &failure.cause else {
        panic!("expected a definition type error, got: {}", failure.cause);
    };
    assert_eq!(definition, "implicit-2d-flatten");
}
