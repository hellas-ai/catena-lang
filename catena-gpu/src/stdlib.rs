use std::path::{Path, PathBuf};

/// One source file embedded in the minimal standard library.
pub struct StdlibFile {
    pub filename: &'static str,
    pub source: &'static str,
}

macro_rules! stdlib_files {
    ($($filename:literal),+ $(,)?) => {
        &[$(
            StdlibFile {
                filename: $filename,
                source: include_str!(concat!("../stdlib/", $filename)),
            },
        )+]
    };
}

/// The complete, deliberately small standard library used by `catena-gpu`.
pub const FILES: &[StdlibFile] = stdlib_files![
    "value.hex",
    "data.hex",
    "buf.hex",
    "fn.hex",
    "index.hex",
    "product.hex",
    "combinators.hex",
    "gpu.hex",
    "matrix.hex",
    "gpu-matrix.hex",
    "gpu-matmul.hex",
    "gpu-tiled-matmul.hex",
];

pub fn sources() -> impl ExactSizeIterator<Item = &'static str> {
    FILES.iter().map(|file| file.source)
}

pub fn paths_from(root: impl AsRef<Path>) -> impl ExactSizeIterator<Item = PathBuf> {
    let stdlib = root.as_ref().join("stdlib");
    FILES.iter().map(move |file| stdlib.join(file.filename))
}
