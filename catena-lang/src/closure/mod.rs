//! Closure conversion
//!
//! Given an AnnotatedTerm f containing a node n with closure type A => B,
//! identify its "smuggled environment" X, and produce:
//!
//! 1. A new term `closure.f.n : X ● A -> B`
//! 2. A modified `f` whose closure is replaced by `closure.f.n`

// Identifies a region of an AnnotatedTerm corresponding to a (bounded) closure
pub mod region;

// Create an AnnotatedTerm `t` corresponding to a subregion identified by `region`
pub mod extract;

// From the extracted AnnotatedTerm create `closure.f.n` as `(f × id_A) ; evaluate`,
// where `evaluate` is the (inlined) body of cmc.hex's evaluate.
pub mod closure;
