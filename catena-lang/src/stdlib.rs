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

/// Checked-in experimental library sources in manifest order.
///
/// This library is standalone. Selecting a code generator does not load it;
/// callers combine these sources with their program and select
/// [`crate::codegen::CodegenKind::Experimental`] explicitly.
pub fn experimental_sources() -> impl ExactSizeIterator<Item = &'static str> {
    EXPERIMENTAL_SOURCES.iter().copied()
}

const EXPERIMENTAL_SOURCES: &[&str] = &[
    include_str!("../stdlib/experimental/metacat/cmc.hex"),
    include_str!("../stdlib/experimental/metacat/fn.hex"),
    include_str!("../stdlib/experimental/metacat/literals.hex"),
    include_str!("../stdlib/experimental/metacat/product.hex"),
    include_str!("../stdlib/experimental/metacat/value.hex"),
    include_str!("../stdlib/experimental/metacat/witnesses.hex"),
    include_str!("../stdlib/experimental/assert.hex"),
    include_str!("../stdlib/experimental/equality.hex"),
    include_str!("../stdlib/experimental/fold.hex"),
    include_str!("../stdlib/experimental/gpu.hex"),
    include_str!("../stdlib/experimental/lists.hex"),
    include_str!("../stdlib/experimental/numeric.hex"),
    include_str!("../stdlib/experimental/prelude.hex"),
    include_str!("../stdlib/experimental/runtime.hex"),
    include_str!("../stdlib/experimental/unsafe.hex"),
    include_str!("../stdlib/experimental/gpu/barriers.hex"),
    include_str!("../stdlib/experimental/gpu/geometry.hex"),
    include_str!("../stdlib/experimental/gpu/launch.hex"),
    include_str!("../stdlib/experimental/gpu/matmul.hex"),
    include_str!("../stdlib/experimental/gpu/memory.hex"),
    include_str!("../stdlib/experimental/gpu/protocol.hex"),
    include_str!("../stdlib/experimental/execution.hex"),
];

pub fn paths_from(root: impl AsRef<Path>) -> impl ExactSizeIterator<Item = PathBuf> {
    let stdlib = root.as_ref().join("stdlib/default");
    FILES.iter().map(move |file| stdlib.join(file.filename))
}
