use super::*;

const PAST_END_PERMISSION: &str = include_str!("permissions/past_end_cell.hex");

#[test]
fn permission_query_rejects_a_past_end_buffer_cell() {
    let raw = RawTheorySet::from_texts(stdlib::sources().chain([PAST_END_PERMISSION])).unwrap();
    let failure = compile(raw).expect_err("an unbounded u64 must not be accepted as a cell index");

    let CompileError::Check(CheckError::Definition {
        definition, error, ..
    }) = &failure.cause
    else {
        panic!("expected a definition type error, got: {}", failure.cause);
    };
    assert_eq!(definition, "permission-for-past-end-cell-kernel");

    let reason = format!("{error:?}");
    assert!(
        reason.contains("MatchError")
            && reason.contains("Operation(\"u64\"),")
            && reason.contains(r#"Operation(\"ix\")"#),
        "expected an unbounded-u64 versus bounded-index mismatch: {reason}"
    );
}
