//! Catena GPU Runtime

/// Public API for creating values to pass into generated catena code
pub mod value;

/// Helpers for creating and freeing Catena memory values on program boundaries
pub mod mem;

/// Explicit GPU allocations and legacy cross-process IPC mappings.
pub mod device_mem;

/// Compile generated GPU C++ to a shared object.
pub mod artifact;

/// Marshal catena values into the C ABI and invoke compiled symbols
pub(crate) mod executor;

/// manage and run compiled catena programs
pub mod runtime;

/// Runtime-call signature metadata.
pub mod signature;

//#[cfg(test)]
//mod tests;

pub use device_mem::{DeviceAllocator, DeviceBuffer, IpcMemoryHandle};
pub use mem::{Mem, MemError};
pub use runtime::{ExecError, InitError, Runtime};
pub use value::Value;
pub use value::ValueKind;
