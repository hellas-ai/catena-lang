use catena_gpu::{
    check::CheckError,
    codegen::{GpuDialect, gpu::render_modules},
    compile::{CompileError, compile},
    stdlib,
};
use metacat::theory::RawTheorySet;

#[path = "compile/basic.rs"]
mod basic;
#[path = "compile/permissions.rs"]
mod permissions;
