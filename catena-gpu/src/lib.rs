//! A minimal Catena-to-GPU compiler and runtime.
//!
//! The included language surface is limited to name elaboration, products,
//! scalar operations, first-order row-major matrix indexing, and a small typed
//! model of grids, scheduling, permissions, and global buffers. Closures/CMC,
//! materialization, closure-based reduction, shared memory, and closure
//! conversion are not part of [`stdlib`].

pub mod check;
pub mod codegen;
pub mod compile;
pub mod elaborate;
mod gpu;
pub mod runtime;
pub mod stdlib;

pub mod report;
