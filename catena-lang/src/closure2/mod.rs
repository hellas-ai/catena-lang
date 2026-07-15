//! Closure conversion over graphs produced by `forget_closures`.
//!
//! The conversion is deliberately split into three stages: discover a delimited
//! control-flow region, turn that region into a definition, and replace the
//! original region with an explicit environment and function pointer.

/// Find regions by following closure domains to their codomains.
pub mod region;

/// Turn discovered regions into `closure.*` definitions and `name.closure.*` declarations.
pub mod definition;

// Future stages:
// - Replace each region with an environment, function pointer, and context operations.
