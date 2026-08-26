//! Compile and execute scalar Catena programs.

mod artifact;
mod executor;
mod signature;
mod value;

pub use artifact::{Artifact, ArtifactError};
pub use runtime::{ExecError, InitError, Runtime};
pub use value::{Value, ValueKind};

mod runtime;
