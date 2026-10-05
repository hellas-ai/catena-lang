//! Experimental code generator.
//!
//! This backend is registered separately from `default`; its lowering and
//! rendering will live here. Until implemented, selection fails explicitly.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CodegenError {
    #[error("experimental codegen is not implemented yet")]
    NotImplemented,
}
