use std::{fs, process::Command};

#[test]
fn cli_writes_only_the_selected_dialect() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let input = directory.path().join("empty.hex");
    fs::write(&input, "")?;
    for backend in ["default", "experimental"] {
        for dialect in ["hip", "cuda"] {
            let output_dir = directory.path().join(format!("{backend}-{dialect}"));
            let mut command = Command::new(env!("CARGO_BIN_EXE_catena-lang"));
            command.arg(&input).arg("-o").arg(&output_dir).args([
                "--no-stdlib",
                "--no-svg",
                "--codegen",
                backend,
            ]);
            // HIP is also the CLI default.
            if dialect == "cuda" {
                command.args(["--dialect", dialect]);
            }
            let output = command.output()?;
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let files = fs::read_dir(output_dir.join("gpu"))?
                .map(|entry| entry.map(|entry| entry.file_name()))
                .collect::<Result<Vec<_>, _>>()?;
            assert_eq!(files, [std::ffi::OsString::from(format!("{dialect}.cpp"))]);
            let source = fs::read_to_string(output_dir.join("gpu").join(&files[0]))?;
            let header = if dialect == "hip" {
                "hip/hip_runtime.h"
            } else {
                "cuda_runtime.h"
            };
            assert!(source.contains(header));
            assert!(output_dir.join("raw_theories.hex").exists());
        }
    }
    Ok(())
}

#[test]
fn cli_preserves_diagnostics_on_compile_failure() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let input = directory.path().join("invalid.hex");
    fs::write(&input, "(def program bad : [] -> [] = missing)")?;
    let output_dir = directory.path().join("report");
    let output = Command::new(env!("CARGO_BIN_EXE_catena-lang"))
        .arg(&input)
        .arg("-o")
        .arg(&output_dir)
        .args(["--no-stdlib", "--no-svg"])
        .output()?;
    assert!(!output.status.success());
    assert!(output_dir.join("raw_theories.hex").exists());
    assert!(!output_dir.join("gpu").exists());
    Ok(())
}
