use super::*;

const MISSING_PROOF: &str = include_str!("numeric/missing_proof.hex");
const NAIVE_U64_MATMUL: &str = include_str!("../cases/matmul/naive_u64.hex");

#[test]
fn numeric_evidence_specializes_generic_matmul() {
    let raw = RawTheorySet::from_texts(stdlib::sources().chain([NAIVE_U64_MATMUL])).unwrap();
    let report = compile(raw).unwrap();
    let generated = render_modules(report.gpu_modules.as_ref().unwrap(), GpuDialect::Hip).unwrap();

    assert!(generated.contains(" = 0;"));
    assert!(generated.contains(" * "));
    assert!(generated.contains(" + "));
    assert!(!generated.contains("gpu_numeric"));
    assert!(!generated.contains("template<"));
    assert!(!generated.contains("typename T"));
    assert!(generated.contains("program_gpu_global_matrix_matmul_kernel__"));
}

#[test]
fn generic_scalar_arithmetic_requires_numeric_evidence() {
    let raw = RawTheorySet::from_texts(stdlib::sources().chain([MISSING_PROOF])).unwrap();
    let failure = compile(raw).expect_err("generic scalar arithmetic needs gpu.numeric evidence");

    let CompileError::Check(CheckError::Definition { definition, .. }) = &failure.cause else {
        panic!("expected a definition type error, got: {}", failure.cause);
    };
    assert_eq!(definition, "unsupported-without-numeric-proof");
}
