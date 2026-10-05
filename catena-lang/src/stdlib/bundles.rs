use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path, PathBuf},
};

use anyhow::{Context, bail};
use serde::Deserialize;

use super::{BUNDLES, StdlibFile};

/// An embedded source bundle. An empty `extends` list denotes a standalone bundle.
pub struct StdlibBundle {
    pub name: &'static str,
    pub extends: &'static [&'static str],
    pub files: &'static [StdlibFile],
}

/// A source with its filename retained for diagnostics.
#[derive(Clone, Debug)]
pub struct SourceFile {
    pub filename: PathBuf,
    pub source: Cow<'static, str>,
}

struct Bundle {
    extends: Vec<String>,
    files: Vec<SourceFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    name: String,
    #[serde(default)]
    extends: Vec<String>,
    #[serde(default)]
    files: Option<Vec<PathBuf>>,
}

/// Embedded and explicitly registered local bundles. Local dependencies must
/// be registered with `add_directory` too.
pub struct BundleRegistry {
    bundles: BTreeMap<String, Bundle>,
}

impl Default for BundleRegistry {
    fn default() -> Self {
        Self::new(BUNDLES).expect("embedded stdlib bundle names must be unique and nonempty")
    }
}

impl BundleRegistry {
    pub fn new(bundles: &[StdlibBundle]) -> anyhow::Result<Self> {
        let mut registry = Self {
            bundles: BTreeMap::new(),
        };
        for bundle in bundles {
            registry.insert(
                bundle.name,
                Bundle {
                    extends: bundle.extends.iter().map(|name| (*name).into()).collect(),
                    files: bundle
                        .files
                        .iter()
                        .map(|file| SourceFile {
                            filename: Path::new("<stdlib>").join(bundle.name).join(file.filename),
                            source: Cow::Borrowed(file.source),
                        })
                        .collect(),
                },
            )?;
        }
        Ok(registry)
    }

    /// Read `stdlib.json` and its source files. Without a `files` list, loads all
    /// `.hex` files directly in the directory, sorted by filename. Returns the bundle name
    /// to use as a selection root. Existing names cannot be overridden.
    pub fn add_directory(&mut self, directory: impl AsRef<Path>) -> anyhow::Result<String> {
        let directory = directory.as_ref();
        let path = directory.join("stdlib.json");
        let text = fs::read_to_string(&path)
            .with_context(|| format!("failed to read stdlib manifest {}", path.display()))?;
        let manifest: Manifest = serde_json::from_str(&text)
            .with_context(|| format!("invalid stdlib manifest {}", path.display()))?;
        self.check_name(&manifest.name)
            .with_context(|| format!("invalid stdlib manifest {}", path.display()))?;
        let mut seen = BTreeSet::new();
        let mut files = Vec::new();
        let filenames = match manifest.files {
            Some(files) => files,
            None => {
                let mut files = Vec::new();
                for entry in fs::read_dir(directory).with_context(|| {
                    format!("failed to read stdlib directory {}", directory.display())
                })? {
                    let entry = entry?;
                    let filename = PathBuf::from(entry.file_name());
                    if entry.path().is_file()
                        && filename.extension().is_some_and(|ext| ext == "hex")
                    {
                        files.push(filename);
                    }
                }
                files.sort();
                files
            }
        };
        for file in filenames {
            if file.as_os_str().is_empty()
                || !file
                    .components()
                    .all(|part| matches!(part, Component::Normal(_)))
            {
                bail!(
                    "stdlib manifest {}: source path must be relative without '.' or '..': {}",
                    path.display(),
                    file.display()
                );
            }
            if !seen.insert(file.clone()) {
                bail!(
                    "stdlib manifest {}: repeated source file {}",
                    path.display(),
                    file.display()
                );
            }
            let filename = directory.join(file);
            let source = fs::read_to_string(&filename)
                .with_context(|| format!("failed to read stdlib source {}", filename.display()))?;
            files.push(SourceFile {
                filename,
                source: Cow::Owned(source),
            });
        }
        self.insert(
            &manifest.name,
            Bundle {
                extends: manifest.extends,
                files,
            },
        )?;
        Ok(manifest.name)
    }

    /// Resolve roots in selection order, dependencies in declaration order,
    /// and files in manifest order (or filename order when omitted). Each bundle is included only once. Empty
    /// roots select nothing; callers choose whether to request `default`.
    pub fn resolve(&self, names: &[&str]) -> anyhow::Result<Vec<SourceFile>> {
        let mut visited = BTreeSet::new();
        let mut active = Vec::new();
        let mut files = Vec::new();
        for name in names {
            self.visit(name, &mut visited, &mut active, &mut files)?;
        }
        Ok(files)
    }

    fn check_name(&self, name: &str) -> anyhow::Result<()> {
        if name.is_empty() || name.trim() != name {
            bail!("stdlib bundle name must be nonempty without surrounding whitespace");
        }
        if self.bundles.contains_key(name) {
            bail!("duplicate stdlib bundle name `{name}`");
        }
        Ok(())
    }

    fn insert(&mut self, name: &str, bundle: Bundle) -> anyhow::Result<()> {
        self.check_name(name)?;
        self.bundles.insert(name.into(), bundle);
        Ok(())
    }

    fn visit<'a>(
        &'a self,
        name: &'a str,
        visited: &mut BTreeSet<&'a str>,
        active: &mut Vec<&'a str>,
        files: &mut Vec<SourceFile>,
    ) -> anyhow::Result<()> {
        if visited.contains(name) {
            return Ok(());
        }
        if active.contains(&name) {
            let mut cycle = active.clone();
            cycle.push(name);
            bail!("stdlib dependency cycle: {}", cycle.join(" -> "));
        }
        let bundle = self.bundles.get(name).with_context(|| {
            let available = self
                .bundles
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join(", ");
            format!("unknown stdlib bundle `{name}` (available: {available})")
        })?;
        active.push(name);
        for dependency in &bundle.extends {
            self.visit(dependency, visited, active, files)
                .with_context(|| format!("stdlib bundle `{name}` requires `{dependency}`"))?;
        }
        active.pop();
        visited.insert(name);
        files.extend(bundle.files.iter().cloned());
        Ok(())
    }
}
