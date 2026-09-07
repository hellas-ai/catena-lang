use super::*;

const SKIPS_DECLARED_BARRIER: &str = include_str!("barriers/kernel_skips_declared_barrier.hex");
const LEAVES_ITERATION_BARRIER_UNCONSUMED: &str =
    include_str!("barriers/kernel_iteration_leaves_barrier_unconsumed.hex");
const SYNC_USES_WRONG_PRECONDITION: &str =
    include_str!("barriers/sync_rejects_wrong_precondition.hex");
const SYNC_CLAIMS_WRONG_POSTCONDITION: &str =
    include_str!("barriers/sync_rejects_wrong_postcondition.hex");
const SYNC_USES_OWNERSHIP_FOR_WRONG_SHARED_CELL: &str =
    include_str!("barriers/sync_rejects_wrong_shared_cell.hex");
const SYNC_USES_OWNERSHIP_FROM_WRONG_PHASE: &str =
    include_str!("barriers/sync_rejects_ownership_from_wrong_phase.hex");

#[test]
fn kernel_that_skips_a_declared_barrier_is_rejected() {
    assert_barrier_definition_is_rejected(
        SKIPS_DECLARED_BARRIER,
        "kernel-that-skips-declared-barrier",
    );
}

#[test]
fn kernel_iteration_that_leaves_a_barrier_unconsumed_is_rejected() {
    assert_barrier_definition_is_rejected(
        LEAVES_ITERATION_BARRIER_UNCONSUMED,
        "kernel-iteration-that-leaves-barrier-unconsumed",
    );
}

#[test]
fn sync_that_uses_the_wrong_precondition_is_rejected() {
    assert_barrier_definition_is_rejected(
        SYNC_USES_WRONG_PRECONDITION,
        "sync-that-uses-the-wrong-precondition",
    );
}

#[test]
fn sync_that_claims_the_wrong_postcondition_is_rejected() {
    assert_barrier_definition_is_rejected(
        SYNC_CLAIMS_WRONG_POSTCONDITION,
        "sync-that-claims-the-wrong-postcondition",
    );
}

#[test]
fn sync_that_uses_ownership_for_the_wrong_shared_cell_is_rejected() {
    assert_barrier_definition_is_rejected(
        SYNC_USES_OWNERSHIP_FOR_WRONG_SHARED_CELL,
        "sync-that-uses-ownership-for-the-wrong-shared-cell",
    );
}

#[test]
fn sync_that_uses_ownership_from_the_wrong_phase_is_rejected() {
    assert_barrier_definition_is_rejected(
        SYNC_USES_OWNERSHIP_FROM_WRONG_PHASE,
        "sync-that-uses-ownership-from-the-wrong-phase",
    );
}

fn assert_barrier_definition_is_rejected(source: &'static str, expected_definition: &str) {
    let raw = RawTheorySet::from_texts(stdlib::sources().chain([source])).unwrap();
    let failure = compile(raw).expect_err("barrier-unsafe definition must not compile");

    let CompileError::Check(CheckError::Definition { definition, .. }) = &failure.cause else {
        panic!("expected a definition type error, got: {}", failure.cause);
    };
    assert_eq!(definition, expected_definition);
}
