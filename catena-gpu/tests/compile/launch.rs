use super::*;

const UNREFINED_GRID_EXTENT: &str = include_str!("launch/unrefined_grid_extent.hex");

#[test]
fn grid_construction_requires_positive_u32_extents() {
    let raw = RawTheorySet::from_texts(stdlib::sources().chain([UNREFINED_GRID_EXTENT])).unwrap();
    let failure = compile(raw).expect_err("a raw u32 must not be accepted as a launch extent");

    let CompileError::Check(CheckError::Definition { definition, .. }) = &failure.cause else {
        panic!("expected a definition type error, got: {}", failure.cause);
    };
    assert_eq!(definition, "unrefined-grid-extent");
}
