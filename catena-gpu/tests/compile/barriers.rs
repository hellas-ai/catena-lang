use super::*;

const SKIPS_DECLARED_BARRIER: &str = include_str!("barriers/kernel_skips_declared_barrier.hex");
const LEAVES_ITERATION_BARRIER_UNCONSUMED: &str =
    include_str!("barriers/kernel_iteration_leaves_barrier_unconsumed.hex");

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

fn assert_barrier_definition_is_rejected(source: &'static str, expected_definition: &str) {
    let raw = RawTheorySet::from_texts(stdlib::sources().chain([source])).unwrap();
    let failure = compile(raw).expect_err("barrier-unsafe definition must not compile");

    let CompileError::Check(CheckError::Definition { definition, .. }) = &failure.cause else {
        panic!("expected a definition type error, got: {}", failure.cause);
    };
    assert_eq!(definition, expected_definition);
}
