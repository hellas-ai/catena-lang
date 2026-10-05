//! Integration tests for compiler phases.
//!
//! These tests deliberately use the public [`catena_lang::compile::compile`]
//! entry point, rather than invoking closure conversion in isolation. This
//! keeps elaboration, checking, closure-boundary inlining, `forget_closures`,
//! closure conversion, and product lowering in scope.
//!
//! Most tests stop observing the report at `unpacked_products`; one focused boundary test
//! also verifies that compiler output can be rendered as a runtime module. This suite does
//! not create a runtime or execute a program; runtime behavior belongs in `runtime.rs`.

use catena_lang::{
    codegen,
    compile::{CompileError, compile, compile_with_codegen},
    report::{CompileReport, ReportOptions},
    runtime::{GpuDialect, ValueKind},
};
use metacat::theory::RawTheorySet;

#[path = "compile/support.rs"]
mod support;

#[path = "compile/closures/mod.rs"]
mod closures;

/// Compile user sources through the same public entry point used by clients and
/// return the complete phase report for structural assertions.
fn compile_with_sources(
    sources: impl IntoIterator<Item = &'static str>,
) -> anyhow::Result<CompileReport> {
    let raw = RawTheorySet::from_texts(catena_lang::stdlib::sources().chain(sources))?;
    compile(raw).map_err(Into::into)
}

/// Require the public pipeline to complete closure conversion, while allowing
/// an unrelated failure in a later lowering or code-generation phase.
fn compile_through_closure_conversion_with_sources(
    sources: impl IntoIterator<Item = &'static str>,
) -> anyhow::Result<CompileReport> {
    let raw = RawTheorySet::from_texts(catena_lang::stdlib::sources().chain(sources))?;
    match compile(raw) {
        Ok(report) => Ok(report),
        Err(failure) if failure.report.closure_conversion.is_some() => Ok(failure.report),
        Err(failure) => Err(failure.into()),
    }
}

#[test]
fn codegen_produces_a_runtime_module() -> anyhow::Result<()> {
    let raw = RawTheorySet::from_texts(
        catena_lang::stdlib::sources()
            .chain(["(def program identity : (u64 val) -> (u64 val) = [value])"]),
    )?;
    let report = compile(raw)?;
    let modules = report
        .gpu_modules
        .as_ref()
        .expect("successful compilation should contain generated modules");
    assert_eq!(modules.kind(), codegen::CodegenKind::Default);
    let directory = tempfile::tempdir()?;
    report.dump_to_dir_with_options(
        directory.path(),
        ReportOptions {
            #[cfg(feature = "svg-reports")]
            generate_svgs: false,
        },
    )?;

    for (dialect, filename) in [(GpuDialect::Hip, "hip.cpp"), (GpuDialect::Cuda, "cuda.cpp")] {
        let module = codegen::runtime_module(modules, dialect)?;
        assert_eq!(module.dialect, dialect);
        assert!(!module.source.is_empty());
        assert_eq!(
            std::fs::read_to_string(directory.path().join("gpu").join(filename))?,
            module.source,
        );
        let identity = module
            .functions
            .iter()
            .find(|function| function.source_name == "identity")
            .expect("identity should be exported");
        assert_eq!(identity.inputs, [ValueKind::U64]);
        assert_eq!(identity.outputs, [ValueKind::U64]);
    }
    Ok(())
}

#[test]
fn experimental_codegen_reports_unavailable_without_default_fallback() -> anyhow::Result<()> {
    let failure = compile_with_codegen(
        RawTheorySet::from_text("")?,
        codegen::CodegenKind::Experimental,
    )
    .unwrap_err();
    assert!(matches!(
        failure.cause,
        CompileError::Codegen(codegen::CodegenError::Experimental(
            codegen::experimental::CodegenError::NotImplemented
        ))
    ));
    assert!(failure.report.elaborated.is_none());
    assert!(failure.report.gpu_modules.is_none());
    let error = codegen::codegen(
        codegen::CodegenKind::Experimental,
        &Default::default(),
        GpuDialect::Hip,
    )
    .unwrap_err();
    assert!(matches!(error, codegen::CodegenError::Experimental(_)));
    Ok(())
}
