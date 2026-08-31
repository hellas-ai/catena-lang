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
//!     let mut runtime = Runtime::new(GpuDialect::Hip)?;
//!     let artifact = runtime.load_sources(stdlib::sources().chain([PROGRAM]))?;
//!     let [result] = artifact.exec("add-one", [41_u64.into()])?;
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
//! - [`Runtime::new`] creates a process-local GPU context.
//! - [`Runtime::load`] and [`Runtime::load_sources`] compile programs into an [`Artifact`].
//! - [`Artifact::exec`] calls a program from a compiled artifact with [`Value`] inputs.
//! - [`Runtime::mem_u16`], [`Runtime::mem_u64`], and [`Runtime::mem_f32`] copy host slices into owned device memory.
//! ### [`Value`] and Memory
//!
//! Values are input to a catena program by supplying [`Value`]s to [`Artifact::exec`].
//! In addition to scalars like [`Value::U64`], you can supply two kinds of memory:
//! [`MemRef`] and [`MemOwn`].
//!
//! Both are *length-tagged device byte pointers* with differing ownership semantics:
//!
//! - [`MemRef`]: A reference; ownership retained by Rust
//! - [`MemOwn`]: An *owned* buffer: ownership is *transferred to the catena program*
//!
//! Raw device pointers can be wrapped with unsafe [`MemOwn::from_raw_parts`] or
//! [`MemRef::from_raw_parts`], depending on ownership. For example, use [`MemRef::from_raw_parts`]
//! with a local lease for an imported GPU IPC pointer whose memory is managed by another process.

/// Public API for creating values to pass into generated catena code
pub mod value;

/// Helpers for creating and freeing Catena memory values on program boundaries
pub mod mem;

/// Compile and run catena programs
pub mod runtime;

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
pub use runtime::{Artifact, ExecError, InitError, Runtime};
#[cfg(feature = "experimental-catena-gpu")]
pub use signature::GeneratedFunction;
pub use value::Value;
pub use value::ValueKind;
