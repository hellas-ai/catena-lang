//! Integration tests for compiler phases.
//!
//! These tests deliberately use the public [`catena_lang::compile::compile`]
//! entry point, rather than invoking closure conversion in isolation. This
//! keeps elaboration, checking, closure-boundary inlining, `forget_closures`,
//! closure conversion, and product lowering in scope.
//!
//! Most tests stop observing the report at `unpacked_products`; one focused boundary test
//! also verifies that compiler output is a runtime module. This suite does
//! not create a runtime or execute a program; runtime behavior belongs in `runtime.rs`.

use catena_lang::{
    codegen,
    compile::compile,
    report::CompileReport,
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
    let mut report = CompileReport::new(raw);
    compile(&mut report, codegen::CodegenKind::Default, GpuDialect::Hip)?;
    Ok(report)
}

/// Require the public pipeline to complete closure conversion, while allowing
/// an unrelated failure in a later lowering or code-generation phase.
fn compile_through_closure_conversion_with_sources(
    sources: impl IntoIterator<Item = &'static str>,
) -> anyhow::Result<CompileReport> {
    let raw = RawTheorySet::from_texts(catena_lang::stdlib::sources().chain(sources))?;
    let mut report = CompileReport::new(raw);
    match compile(&mut report, codegen::CodegenKind::Default, GpuDialect::Hip) {
        Ok(_) => Ok(report),
        Err(_) if report.closure_conversion.is_some() => Ok(report),
        Err(error) => Err(error.into()),
    }
}

#[test]
fn codegen_produces_a_runtime_module() -> anyhow::Result<()> {
    let raw = RawTheorySet::from_texts(
        catena_lang::stdlib::sources()
            .chain(["(def program identity : (u64 val) -> (u64 val) = [value])"]),
    )?;
    for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
        let mut report = CompileReport::new(raw.clone());
        let module = compile(&mut report, codegen::CodegenKind::Default, dialect)?;
        assert_eq!(module.dialect, dialect);
        assert!(!module.source.is_empty());
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
fn experimental_codegen_uses_shared_passes_and_dispatch() -> anyhow::Result<()> {
    let mut report = CompileReport::new(RawTheorySet::from_text("")?);
    let module = compile(
        &mut report,
        codegen::CodegenKind::Experimental,
        GpuDialect::Hip,
    )?;
    assert_eq!(module.dialect, GpuDialect::Hip);
    assert!(report.closure_conversion.is_some());
    assert!(report.unpacked_products.is_some());
    assert!(module.functions.is_empty());
    Ok(())
}
