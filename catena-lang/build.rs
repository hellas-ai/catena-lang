use std::{
    collections::BTreeSet,
    env, fs,
    path::{Component, PathBuf},
};

fn main() {
    compiler_fingerprint();
    embed_stdlib_sources().unwrap_or_else(|error| panic!("{error}"));
}

fn embed_stdlib_sources() -> Result<(), String> {
    let directory =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("stdlib/default");
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
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("stdlib_files.rs");
    fs::write(&output, sources).map_err(|error| format!("write {}: {error}", output.display()))
}

// Invalidate persistent modules on compiler, stdlib, dependency, or feature changes,
// including edits that have not been committed. Paths are relative for reproducibility.
fn compiler_fingerprint() {
    use sha2::{Digest, Sha256};
    fn walk(path: &std::path::Path, paths: &mut Vec<PathBuf>) {
        if path.is_dir() {
            println!("cargo:rerun-if-changed={}", path.display());
            for entry in fs::read_dir(path).unwrap() {
                walk(&entry.unwrap().path(), paths);
            }
        } else {
            paths.push(path.to_owned());
        }
    }
    fn part(hash: &mut Sha256, bytes: &[u8]) {
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let mut paths = vec![
        root.join("Cargo.toml"),
        root.join("build.rs"),
        root.join("../Cargo.toml"),
        root.join("../Cargo.lock"),
    ];
    walk(&root.join("src"), &mut paths);
    walk(&root.join("stdlib"), &mut paths);
    paths.sort();
    let mut hash = Sha256::new();
    for path in paths {
        part(
            &mut hash,
            path.strip_prefix(&root)
                .unwrap()
                .to_str()
                .unwrap()
                .as_bytes(),
        );
        match fs::read(&path) {
            Ok(bytes) => {
                // Watch only inputs that exist. Cargo reruns a build script
                // whose rerun-if-changed path is missing on every build, and
                // the workspace ../Cargo.{toml,lock} are absent when this crate
                // is vendored, e.g. by Nix.
                println!("cargo:rerun-if-changed={}", path.display());
                part(&mut hash, b"present");
                part(&mut hash, &bytes);
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => part(&mut hash, b"missing"),
            Err(e) => panic!("fingerprint {}: {e}", path.display()),
        }
    }
    let mut configuration = env::vars()
        .filter(|(k, _)| k.starts_with("CARGO_FEATURE_") || k == "TARGET")
        .collect::<Vec<_>>();
    configuration.sort();
    for (k, v) in configuration {
        part(&mut hash, k.as_bytes());
        part(&mut hash, v.as_bytes());
    }
    println!(
        "cargo:rustc-env=CATENA_COMPILER_FINGERPRINT={:x}",
        hash.finalize()
    );
}
