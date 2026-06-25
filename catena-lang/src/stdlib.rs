use std::path::{Path, PathBuf};

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
