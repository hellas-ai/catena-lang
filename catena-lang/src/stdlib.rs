use std::path::{Path, PathBuf};

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

    pub const NAME_PREFIX: &str = "name.";
}

pub struct StdlibFile {
    pub filename: &'static str,
    pub source: &'static str,
}

// Keep the stdlib order in one literal list. `include_str!` cannot consume a
// runtime filename value, so the macro turns each filename literal into both the
// public filename and the matching embedded source.
macro_rules! stdlib_files {
    ($($filename:literal),+ $(,)?) => {
        &[
            $(
                StdlibFile {
                    filename: $filename,
                    source: include_str!(concat!("../stdlib/", $filename)),
                },
            )+
        ]
    };
}

pub const FILES: &[StdlibFile] = stdlib_files![
    "cmc.hex",
    "value.hex",
    "buf.hex",
    "index.hex",
    "data.hex",
    "fn.hex",
    "product.hex",
    "combinators.hex",
    "gpu.hex",
];

pub fn sources() -> impl ExactSizeIterator<Item = &'static str> {
    FILES.iter().map(|file| file.source)
}

pub fn paths_from(root: impl AsRef<Path>) -> impl ExactSizeIterator<Item = PathBuf> {
    let stdlib = root.as_ref().join("stdlib");
    FILES.iter().map(move |file| stdlib.join(file.filename))
}
