//! Integration tests for compiler phases.
//!
//! These tests deliberately use the public [`catena_lang::compile::compile`]
//! entry point, rather than invoking closure conversion in isolation. This
//! keeps elaboration, checking, closure-boundary inlining, `forget_closures`,
//! closure conversion, and product lowering in scope.
//!
//! `compile` also constructs codegen artifacts as its final step, but this test
//! suite stops observing the report at `unpacked_products`: it neither renders
//! generated code nor creates a runtime or executes a program. Runtime behavior
//! belongs in `runtime.rs`.

use catena_lang::{compile::compile, report::CompileReport};
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
