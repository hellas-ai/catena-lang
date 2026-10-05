use std::{env, fs, path::PathBuf};

fn main() {
    let directory =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("stdlib/default");
    println!("cargo:rerun-if-changed={}", directory.display());
    let mut paths = fs::read_dir(&directory)
        .expect("read default stdlib directory")
        .map(|entry| entry.expect("read stdlib entry").path())
        .filter(|path| path.is_file() && path.extension().is_some_and(|ext| ext == "hex"))
        .collect::<Vec<_>>();
    paths.sort();
    let mut sources = String::from("&[\n");
    for path in paths {
        sources.push_str(&format!(
            "StdlibFile {{ filename: {:?}, source: include_str!({:?}) }},\n",
            path.file_name().unwrap().to_str().unwrap(),
            path.to_str().unwrap(),
        ));
    }
    sources.push_str("]\n");
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("stdlib_files.rs");
    fs::write(output, sources).expect("write embedded stdlib file list");
}
