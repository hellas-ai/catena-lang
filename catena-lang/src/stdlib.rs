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
    EXPERIMENTAL_FILES.iter().map(|file| file.source)
}

/// Experimental primitives and core helpers, without the optional matmul helpers.
///
/// Use this for programs implementing their own tensor kernels. The full
/// [`experimental_sources`] library additionally includes `gpu/matmul.hex`.
pub fn experimental_core_sources() -> impl Iterator<Item = &'static str> {
    EXPERIMENTAL_FILES
        .iter()
        .filter(|file| file.filename != "gpu/matmul.hex")
        .map(|file| file.source)
}

const EXPERIMENTAL_FILES: &[StdlibFile] = &[
    StdlibFile {
        filename: "metacat/cmc.hex",
        source: include_str!("../stdlib/experimental/metacat/cmc.hex"),
    },
    StdlibFile {
        filename: "metacat/fn.hex",
        source: include_str!("../stdlib/experimental/metacat/fn.hex"),
    },
    StdlibFile {
        filename: "metacat/literals.hex",
        source: include_str!("../stdlib/experimental/metacat/literals.hex"),
    },
    StdlibFile {
        filename: "metacat/product.hex",
        source: include_str!("../stdlib/experimental/metacat/product.hex"),
    },
    StdlibFile {
        filename: "metacat/value.hex",
        source: include_str!("../stdlib/experimental/metacat/value.hex"),
    },
    StdlibFile {
        filename: "metacat/witnesses.hex",
        source: include_str!("../stdlib/experimental/metacat/witnesses.hex"),
    },
    StdlibFile {
        filename: "assert.hex",
        source: include_str!("../stdlib/experimental/assert.hex"),
    },
    StdlibFile {
        filename: "equality.hex",
        source: include_str!("../stdlib/experimental/equality.hex"),
    },
    StdlibFile {
        filename: "fold.hex",
        source: include_str!("../stdlib/experimental/fold.hex"),
    },
    StdlibFile {
        filename: "gpu.hex",
        source: include_str!("../stdlib/experimental/gpu.hex"),
    },
    StdlibFile {
        filename: "lists.hex",
        source: include_str!("../stdlib/experimental/lists.hex"),
    },
    StdlibFile {
        filename: "numeric.hex",
        source: include_str!("../stdlib/experimental/numeric.hex"),
    },
    StdlibFile {
        filename: "prelude.hex",
        source: include_str!("../stdlib/experimental/prelude.hex"),
    },
    StdlibFile {
        filename: "runtime.hex",
        source: include_str!("../stdlib/experimental/runtime.hex"),
    },
    StdlibFile {
        filename: "unsafe.hex",
        source: include_str!("../stdlib/experimental/unsafe.hex"),
    },
    StdlibFile {
        filename: "gpu/barriers.hex",
        source: include_str!("../stdlib/experimental/gpu/barriers.hex"),
    },
    StdlibFile {
        filename: "gpu/geometry.hex",
        source: include_str!("../stdlib/experimental/gpu/geometry.hex"),
    },
    StdlibFile {
        filename: "gpu/launch.hex",
        source: include_str!("../stdlib/experimental/gpu/launch.hex"),
    },
    StdlibFile {
        filename: "gpu/matmul.hex",
        source: include_str!("../stdlib/experimental/gpu/matmul.hex"),
    },
    StdlibFile {
        filename: "gpu/memory.hex",
        source: include_str!("../stdlib/experimental/gpu/memory.hex"),
    },
    StdlibFile {
        filename: "gpu/protocol.hex",
        source: include_str!("../stdlib/experimental/gpu/protocol.hex"),
    },
    StdlibFile {
        filename: "execution.hex",
        source: include_str!("../stdlib/experimental/execution.hex"),
    },
];

pub fn paths_from(root: impl AsRef<Path>) -> impl ExactSizeIterator<Item = PathBuf> {
    let stdlib = root.as_ref().join("stdlib/default");
    FILES.iter().map(move |file| stdlib.join(file.filename))
}
