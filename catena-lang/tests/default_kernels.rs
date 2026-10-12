use catena_lang::{
    codegen::CodegenKind, compile::compile, report::CompileReport, runtime::GpuDialect, stdlib,
};
use metacat::theory::RawTheorySet;

#[test]
fn exploratory_examples_compile_through_default_runtime_module() -> anyhow::Result<()> {
    let raw = RawTheorySet::from_texts(
        stdlib::sources().chain([include_str!("../examples/materializec.hex")]),
    )?;
    for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
        let module = compile(
            &mut CompileReport::new(raw.clone()),
            CodegenKind::Default,
            dialect,
        )?;
        assert_eq!(module.dialect, dialect);
        for name in [
            "materialize-into-f32",
            "materialize-borrow-topk-f32",
            "materialize-bf16-gemm-wmma",
            "materialize-borrow-native-bf16-gemm-wmma-pair",
            "materialize-borrow-cached-attention-bf16",
        ] {
            assert!(
                module.functions.iter().any(|f| f.source_name == name),
                "missing default entry point {name}"
            );
        }
        assert!(!module.source.contains("blas"));
        assert!(
            module
                .source
                .contains("catena_host_execution_scope_t execution_scope;")
        );
        let (alloc, free, sync, mma) = match dialect {
            GpuDialect::Hip => (
                "hipMallocAsync",
                "hipFreeAsync",
                "hipDeviceSynchronize",
                "rocwmma::bfloat16_t",
            ),
            GpuDialect::Cuda => (
                "cudaMallocAsync",
                "cudaFreeAsync",
                "cudaDeviceSynchronize",
                "nvcuda::wmma::fragment",
            ),
        };
        for required in [alloc, free, sync, mma] {
            assert!(module.source.contains(required), "missing {required}");
        }
    }
    Ok(())
}
