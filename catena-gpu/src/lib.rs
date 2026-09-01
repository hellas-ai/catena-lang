//! A minimal Catena-to-GPU compiler using catena-lang's runtime.
//!
//! The included language surface is limited to name elaboration, products,
//! scalar operations, first-order row-major matrix indexing, and a small typed
//! model of grids, scheduling, permissions, and global buffers.

pub mod check;
pub mod codegen;
pub mod compile;
pub mod elaborate;
pub mod runtime;
pub mod stdlib;

pub mod report;
