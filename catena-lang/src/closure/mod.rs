//! Closure conversion
//!
//! Suppose we have a term `f` containing a node `n` with closure type `Args ⇒  Result`.
//! Closure conversion will rewrite this node to have type `Context ● (Context × Args -> Result)`,
//! i.e., a pair of context data and function pointer.
//!
//! This module exports two main things:
//!
//! - The `convert` function is the low level, per-definition implementation of this procedure.
//! - The `theory` module closure-converts a whole theory by lowering `if` calls to `ifc`.
//!
//! # How it works
//!
//! We outline this process below, by example.
//! Suppose
//!
//! - `Args = (1×1)×A`
//! - `Context = X₀ × X₁`
//!
//! Closure conversion proceeds as follows:
//!
//! - `region`: From a given closure-typed node, identify the "closure region" inside `f`
//! - `extract`: define a new arrow `extracted : Context -> (Args => Result)`
//! - `body`: construct `body : Context × Args -> Result`
//! - `rewrite`: Splice `body` back into `f`, replacing the original region.
//!
//! Note that both `Context` and `Args` are *packed* objects, meaning they are represented as a
//! *single wire*.
//!

// Identifies a region of an AnnotatedTerm corresponding to a closure.
pub(crate) mod region;

// Create an AnnotatedTerm `t` corresponding to a subregion identified by the `region` module
pub(crate) mod extract;

// From the extracted AnnotatedTerm create `closure.f.n` as `(f × (unpack {defer .. defer}) ; compose ; run`
// Note that the *outermost* type is always arity-2 `X ● A -> B`.
// (see also `cmc.hex`'s "evaluate" function)
pub(crate) mod body;

// Replace an identified closure region with a caller-provided lowered body.
// NOTE: this module is really doing rewriting internally.
// In future, we should integrate using 'proper' rewrite machinery, rather than reimplementing it
// ad-hoc here.
pub(crate) mod rewrite;

// Integrate region identification, extraction, body construction, and splicing.
pub mod convert;

// Convert a whole theory: for each arrow, identify the closure nodes appearing as (closure) inputs
// to a primitive (`if`), and closure-convert them.
pub mod theory;

#[cfg(test)]
mod tests;
