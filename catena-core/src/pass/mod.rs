//! # compiler passes
//!
//! Each pass is an *endofunctor* on the category of syntax.
//! Lowering is the composition of these functors.
pub mod erase;
pub mod expand_eta;
pub mod forget_bound;
pub mod forget_loopback;
pub mod inline;

pub mod discard_naturality;
