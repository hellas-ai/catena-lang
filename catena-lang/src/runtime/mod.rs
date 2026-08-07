//! Catena GPU Runtime

/// Public API for creating values to pass into generated catena code
pub mod value;

/// Helpers for creating and freeing Catena memory values on program boundaries
pub mod mem;

/// Low-level GPU operations needed to manage Runtime-owned allocations.
mod gpu_api;

/// Compile generated GPU C++ to a shared object.
mod artifact;

/// Marshal catena values into the C ABI and invoke compiled symbols
mod executor;

/// manage and run compiled catena programs
pub mod runtime;

/// Runtime-call signature metadata.
mod signature;

//#[cfg(test)]
//mod tests;

pub use artifact::ArtifactError;
pub use half::bf16;
pub use mem::{MemError, MemOwn, MemRef};
pub use runtime::{ExecError, InitError, Runtime};
pub use value::Value;
pub use value::ValueKind;
