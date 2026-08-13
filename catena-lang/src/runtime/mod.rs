//! # Catena GPU Runtime
//!
//! Load and run catena programs from Rust.
//! Minimal HIP example:
//!
//! ```no_run
//! use catena_lang::{
//!     codegen::GpuDialect,
//!     runtime::{Runtime, Value},
//!     stdlib,
//! };
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let runtime = Runtime::from_sources(stdlib::sources().chain([PROGRAM]), GpuDialect::Hip)?;
//!     let [result] = runtime.exec("add-one", [41_u64.into()])?;
//!     let Value::U64(sum) = result else {
//!         panic!("`add-one` returned an unexpected value: {result:?}");
//!     };
//!     assert_eq!(sum, 42);
//!     Ok(())
//! }
//!
//! const PROGRAM: &str = "(def program add-one : (u64 val) -> (u64 val) = ({_ const.u64.0x0000000000000001} u64.add))";
//! ```
//!
//! ## Quick reference
//!
//! - [`Runtime::new`] and [`Runtime::from_sources`] load programs from paths or source strings.
//! - [`Runtime::exec`] calls a program with [`Value`] inputs and const-sized outputs.
//! - [`Runtime::mem_u16`], [`Runtime::mem_u64`], and [`Runtime::mem_f32`] copy host slices into owned device memory.
//! ### [`Value`] and Memory
//!
//! Values are input to a catena program by supplying [`Value`]s to [`Runtime::exec`].
//! In addition to scalars like [`Value::U64`], you can supply two kinds of memory:
//! [`MemRef`] and [`MemOwn`].
//!
//! Both are *length tagged device byte pointers* with differing ownership semantics:
//!
//! - [`MemRef`]: A reference; ownership retained by Rust
//! - [`MemOwn`]: A *owned* buffer: ownership is *transferred to the catena program*
//!
//! Raw device pointers can be wrapped with unsafe [`MemOwn::from_raw_parts`] or
//! [`MemRef::from_raw_parts`], depending on ownership. For example, use [`MemRef::from_raw_parts`]
//! with a local lease for an imported GPU IPC pointer whose memory is managed by another process.

/// Public API for creating values to pass into generated catena code
pub mod value;

/// Helpers for creating and freeing Catena memory values on program boundaries
pub mod mem;

/// manage and run compiled catena programs
pub mod runtime;

/// Low-level GPU operations needed to manage Runtime-owned allocations.
pub(crate) mod gpu_api;

/// Compile generated GPU C++ to a shared object.
mod artifact;

/// Marshal catena values into the C ABI and invoke compiled symbols
mod executor;

/// Runtime-call signature metadata.
mod signature;

//#[cfg(test)]
//mod tests;

pub use artifact::ArtifactError;
pub use mem::MemError;
pub use mem::MemOwn;
pub use mem::MemRef;
pub use runtime::Runtime;
pub use runtime::{ExecError, InitError};
pub use value::Value;
pub use value::ValueKind;
