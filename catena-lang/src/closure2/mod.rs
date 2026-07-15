//! Closure conversion over graphs produced by `forget_closures`.
//!
//! The conversion is deliberately split into three stages: discover a delimited
//! control-flow region, turn that region into a definition, and replace the
//! original region with an explicit environment and function pointer.

/// Find regions by following closure domains to their codomains.
pub mod region;

// Future stages:
// - Create one new definition from each discovered closure region.
// - Replace each region with an environment, function pointer, and context operations.
