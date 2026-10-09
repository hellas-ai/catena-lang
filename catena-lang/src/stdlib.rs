use std::path::{Path, PathBuf};

mod bundles;
pub use bundles::{BundleRegistry, SourceFile, StdlibBundle};

/// Names of built in stdlib types and operations
pub mod constants {
    // Type of the internal hom
    pub const FN_HOM_TYPE: &str = "=>";
    // Type of function references (codomain of function *names*)
    pub const FN_REF_TYPE: &str = "->";
    pub const PRODUCT_TYPE: &str = "*";
    pub const UNIT_TYPE: &str = "1";
    pub const VALUE_TYPE: &str = "val";

    pub const PRODUCT_INTRO: &str = "*.intro";
    pub const PRODUCT_ELIM: &str = "*.elim";
    pub const UNIT_INTRO: &str = "unit.intro";
    pub const UNIT_ELIM: &str = "unit.elim";

    pub const DEFER: &str = "defer";
    pub const RUN: &str = "run";
    pub const COMPOSE: &str = "compose";
    pub const TENSOR: &str = "tensor";
    pub const LIFT: &str = "lift";
    pub const EVAL: &str = "eval";
}

pub struct StdlibFile {
    pub filename: &'static str,
    pub source: &'static str,
}

/// All `.hex` files in `stdlib/default`, embedded in filename order at build time.
pub const FILES: &[StdlibFile] = include!(concat!(env!("OUT_DIR"), "/stdlib_files.rs"));

/// Embedded bundles available by name. Dependencies are loaded before files.
pub const BUNDLES: &[StdlibBundle] = &[StdlibBundle {
    name: "default",
    extends: &[],
    files: FILES,
}];

pub fn sources() -> impl ExactSizeIterator<Item = &'static str> {
    FILES.iter().map(|file| file.source)
}

pub fn paths_from(root: impl AsRef<Path>) -> impl ExactSizeIterator<Item = PathBuf> {
    let stdlib = root.as_ref().join("stdlib/default");
    FILES.iter().map(move |file| stdlib.join(file.filename))
}

/// Standalone experimental library, embedded in manifest order.
/// Select experimental codegen separately; these sources are not the default library.
pub fn experimental_sources() -> impl ExactSizeIterator<Item = &'static str> {
    EXPERIMENTAL_FILES.iter().map(|file| file.source)
}

/// Experimental primitives and core helpers, excluding optional matmul helpers.
pub fn experimental_core_sources() -> impl Iterator<Item = &'static str> {
    EXPERIMENTAL_FILES
        .iter()
        .filter(|file| file.filename != "gpu/matmul.hex")
        .map(|file| file.source)
}

const EXPERIMENTAL_FILES: &[StdlibFile] =
    include!(concat!(env!("OUT_DIR"), "/experimental_stdlib_files.rs"));
