use std::fs;

use catena_lang::stdlib::{self, BundleRegistry, StdlibBundle, StdlibFile};

#[test]
fn default_manifest_loads_the_same_files_as_the_embedded_bundle() -> anyhow::Result<()> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut registry = BundleRegistry::new(&[])?;
    let name = registry.add_directory(root.join("stdlib/default"))?;
    assert_eq!(name, "default");
    let files = registry.resolve(&[&name])?;
    assert_eq!(
        files
            .iter()
            .map(|file| file.source.as_ref())
            .collect::<Vec<_>>(),
        stdlib::sources().collect::<Vec<_>>()
    );
    assert_eq!(
        files
            .iter()
            .map(|file| file.filename.clone())
            .collect::<Vec<_>>(),
        stdlib::paths_from(root).collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn minimal_manifest_loads_all_hex_files_in_filename_order() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    fs::write(directory.path().join("stdlib.json"), r#"{"name":"custom"}"#)?;
    fs::write(directory.path().join("z.hex"), "# z")?;
    fs::write(directory.path().join("a.hex"), "# a")?;
    fs::write(directory.path().join("notes.txt"), "ignored")?;
    fs::create_dir(directory.path().join("nested"))?;
    fs::write(directory.path().join("nested/other.hex"), "ignored")?;
    let mut registry = BundleRegistry::default();
    registry.add_directory(directory.path())?;
    let files = registry.resolve(&["custom"])?;
    assert_eq!(
        files
            .iter()
            .map(|file| file.source.as_ref())
            .collect::<Vec<_>>(),
        ["# a", "# z"]
    );
    Ok(())
}

#[test]
fn default_bundle_preserves_existing_sources() -> anyhow::Result<()> {
    let registry = BundleRegistry::default();
    let files = registry.resolve(&["default", "default"])?;
    assert_eq!(
        files
            .iter()
            .map(|file| file.source.as_ref())
            .collect::<Vec<_>>(),
        stdlib::sources().collect::<Vec<_>>()
    );
    assert!(registry.resolve(&[])?.is_empty());
    Ok(())
}

#[test]
fn dependencies_are_ordered_and_shared_dependencies_loaded_once() -> anyhow::Result<()> {
    let registry = BundleRegistry::new(&[
        StdlibBundle {
            name: "base",
            extends: &[],
            files: &[StdlibFile {
                filename: "base.hex",
                source: "base",
            }],
        },
        StdlibBundle {
            name: "left",
            extends: &["base"],
            files: &[StdlibFile {
                filename: "left.hex",
                source: "left",
            }],
        },
        StdlibBundle {
            name: "right",
            extends: &["base"],
            files: &[StdlibFile {
                filename: "right.hex",
                source: "right",
            }],
        },
        StdlibBundle {
            name: "top",
            extends: &["right", "left"],
            files: &[
                StdlibFile {
                    filename: "z.hex",
                    source: "z",
                },
                StdlibFile {
                    filename: "a.hex",
                    source: "a",
                },
            ],
        },
    ])?;
    let files = registry.resolve(&["top", "base"])?;
    assert_eq!(
        files
            .iter()
            .map(|file| file.source.as_ref())
            .collect::<Vec<_>>(),
        ["base", "right", "left", "z", "a"]
    );
    Ok(())
}

#[test]
fn unknown_names_cycles_and_duplicate_names_are_errors() -> anyhow::Result<()> {
    let bundles = [
        StdlibBundle {
            name: "a",
            extends: &["b"],
            files: &[],
        },
        StdlibBundle {
            name: "b",
            extends: &["a"],
            files: &[],
        },
        StdlibBundle {
            name: "c",
            extends: &["missing"],
            files: &[],
        },
    ];
    let registry = BundleRegistry::new(&bundles)?;
    let cycle = format!("{:#}", registry.resolve(&["a"]).unwrap_err());
    assert!(cycle.contains("a -> b -> a"), "{cycle}");
    let unknown = format!("{:#}", registry.resolve(&["c"]).unwrap_err());
    assert!(unknown.contains("`c` requires `missing`"), "{unknown}");
    assert!(
        unknown.contains("unknown stdlib bundle `missing`"),
        "{unknown}"
    );
    assert!(registry.resolve(&["missing"]).is_err());
    assert!(
        BundleRegistry::new(&[
            StdlibBundle {
                name: "a",
                extends: &[],
                files: &[]
            },
            StdlibBundle {
                name: "a",
                extends: &[],
                files: &[]
            },
        ])
        .is_err()
    );
    Ok(())
}

#[test]
fn local_manifest_preserves_order_and_selects_only_declared_dependencies() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    fs::write(
        directory.path().join("stdlib.json"),
        r#"{"name":"custom","files":["z.hex","a.hex"]}"#,
    )?;
    fs::write(directory.path().join("z.hex"), "# z")?;
    fs::write(directory.path().join("a.hex"), "# a")?;
    fs::write(directory.path().join("ignored.hex"), "not parsed")?;
    let mut registry = BundleRegistry::default();
    let name = registry.add_directory(directory.path())?;
    let files = registry.resolve(&[&name])?;
    assert_eq!(
        files
            .iter()
            .map(|file| file.source.as_ref())
            .collect::<Vec<_>>(),
        ["# z", "# a"]
    );
    assert_eq!(files[0].filename, directory.path().join("z.hex"));
    assert!(registry.add_directory(directory.path()).is_err());

    fs::write(
        directory.path().join("stdlib.json"),
        r#"{"name":"extension","extends":["default","custom"],"files":[]}"#,
    )?;
    registry.add_directory(directory.path())?;
    assert_eq!(
        registry.resolve(&["extension"])?.len(),
        stdlib::FILES.len() + 2
    );
    Ok(())
}

#[test]
fn invalid_local_manifests_report_the_manifest_or_source_path() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let manifest = directory.path().join("stdlib.json");
    for json in [
        "not json",
        r#"{"name":"custom","files":[],"typo":true}"#,
        r#"{"name":"default","files":[]}"#,
        r#"{"name":"","files":[]}"#,
        r#"{"name":"custom","files":["../outside.hex"]}"#,
        r#"{"name":"custom","files":["/outside.hex"]}"#,
        r#"{"name":"custom","files":[""]}"#,
    ] {
        fs::write(&manifest, json)?;
        let error = format!(
            "{:#}",
            BundleRegistry::default()
                .add_directory(directory.path())
                .unwrap_err()
        );
        assert!(error.contains("stdlib.json"), "{error}");
    }
    fs::write(&manifest, r#"{"name":"custom","files":["missing.hex"]}"#)?;
    let error = format!(
        "{:#}",
        BundleRegistry::default()
            .add_directory(directory.path())
            .unwrap_err()
    );
    assert!(error.contains("missing.hex"), "{error}");
    fs::write(directory.path().join("a.hex"), "")?;
    fs::write(&manifest, r#"{"name":"custom","files":["a.hex","a.hex"]}"#)?;
    let error = format!(
        "{:#}",
        BundleRegistry::default()
            .add_directory(directory.path())
            .unwrap_err()
    );
    assert!(error.contains("repeated source file"), "{error}");
    Ok(())
}
