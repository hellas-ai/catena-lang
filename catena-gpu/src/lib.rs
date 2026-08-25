//! A minimal Catena-to-GPU compiler and runtime.
//!
//! The included language surface is limited to name elaboration, products,
//! scalar values and scalar operations.
//! Closures/CMC, explicit GPU kernels, matrices, buffers, materialization,
//! reduction and source closures are not part of [`stdlib`]. The minimal
//! `bool.ifc` primitive accepts explicit environments and direct function names.

pub mod check;
pub mod codegen;
pub mod compile;
pub mod elaborate;
pub mod runtime;
pub mod stdlib;

pub mod report;
