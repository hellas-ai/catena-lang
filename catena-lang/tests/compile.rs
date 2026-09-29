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
    compile::{compile, compile_sources},
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
fn compile_sources_produces_a_runtime_module() -> anyhow::Result<()> {
    let module = compile_sources(
        catena_lang::stdlib::sources()
            .chain(["(def program identity : (u64 val) -> (u64 val) = [value])"]),
        GpuDialect::Hip,
    )?;

    assert_eq!(module.dialect, GpuDialect::Hip);
    assert!(!module.source.is_empty());
    let identity = module
        .functions
        .iter()
        .find(|function| function.source_name == "identity")
        .expect("identity should be exported");
    assert_eq!(identity.inputs, [ValueKind::U64]);
    assert_eq!(identity.outputs, [ValueKind::U64]);
    Ok(())
}
