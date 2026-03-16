//! Catena C backend

/// Generate C code for a catena program
pub mod codegen;

/// Compile a catena program to a .so file using the C backend
pub mod compile;

/// Marshal catena values into the C ABI and invoke compiled symbols
pub mod executor;

/// manage and run compiled catena programs
pub mod runtime;

pub use runtime::InitError;
pub use runtime::Runtime;
pub use runtime::Value;
