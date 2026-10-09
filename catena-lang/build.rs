use std::{
    collections::BTreeSet,
    env, fs,
    path::{Component, PathBuf},
};

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

    embed_experimental_sources().unwrap_or_else(|error| panic!("{error}"));
}

fn embed_experimental_sources() -> Result<(), String> {
    let directory =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("stdlib/experimental");
    let manifest_path = directory.join("stdlib.json");
    println!("cargo:rerun-if-changed={}", manifest_path.display());
    let text = fs::read_to_string(&manifest_path)
        .map_err(|error| format!("read {}: {error}", manifest_path.display()))?;
    let manifest: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| format!("invalid {}: {error}", manifest_path.display()))?;
    let files: Vec<String> = serde_json::from_value(
        manifest
            .get("files")
            .ok_or_else(|| format!("{}: missing files list", manifest_path.display()))?
            .clone(),
    )
    .map_err(|error| format!("{}: invalid files list: {error}", manifest_path.display()))?;

    let mut seen = BTreeSet::new();
    let mut sources = String::from("&[\n");
    for filename in files {
        let relative = PathBuf::from(&filename);
        if filename.is_empty()
            || !relative
                .components()
                .all(|part| matches!(part, Component::Normal(_)))
        {
            return Err(format!(
                "{}: source path must be relative without '.' or '..': {filename:?}",
                manifest_path.display()
            ));
        }
        if !seen.insert(relative.clone()) {
            return Err(format!(
                "{}: repeated source file {filename:?}",
                manifest_path.display()
            ));
        }
        let path = directory.join(relative);
        println!("cargo:rerun-if-changed={}", path.display());
        if !path
            .metadata()
            .map_err(|error| format!("read source {}: {error}", path.display()))?
            .is_file()
        {
            return Err(format!("source {} is not a file", path.display()));
        }
        sources.push_str(&format!(
            "StdlibFile {{ filename: {filename:?}, source: include_str!({:?}) }},\n",
            path.to_str()
                .ok_or_else(|| format!("source path is not UTF-8: {}", path.display()))?,
        ));
    }
    sources.push_str("]\n");
    let output =
        PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("experimental_stdlib_files.rs");
    fs::write(&output, sources).map_err(|error| format!("write {}: {error}", output.display()))
}
