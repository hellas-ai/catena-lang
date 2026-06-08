use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use super::super::executor::{ArgValue, CallFrame, exec};

#[test]
fn executor_calls_u64_and_f32_symbols() -> Result<(), Box<dyn std::error::Error>> {
    let build_dir = fresh_test_dir();
    fs::create_dir_all(&build_dir)?;

    let c_path = build_dir.join("executor_test.c");
    let so_path = build_dir.join("executor_test.so");
    fs::write(&c_path, test_c_source())?;
    compile_shared_object(&c_path, &so_path)?;

    let x = 7u64;
    let y = 5u64;
    let mut sum = 0u64;
    let mut u64_args = [
        ArgValue::U64(&x),
        ArgValue::U64(&y),
        ArgValue::OutU64(&mut sum),
    ];
    exec(
        &so_path,
        "add_u64",
        CallFrame {
            args: &mut u64_args,
        },
    )?;
    assert_eq!(sum, 12);

    let a = 1.5f32;
    let b = 2.25f32;
    let mut product = 0.0f32;
    let mut f32_args = [
        ArgValue::F32(&a),
        ArgValue::F32(&b),
        ArgValue::OutF32(&mut product),
    ];
    exec(
        &so_path,
        "mul_f32",
        CallFrame {
            args: &mut f32_args,
        },
    )?;
    assert!((product - 3.375).abs() < f32::EPSILON);

    Ok(())
}

fn compile_shared_object(c_path: &Path, so_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let output = Command::new("gcc")
        .arg("-shared")
        .arg("-fPIC")
        .arg(c_path)
        .arg("-o")
        .arg(so_path)
        .output()?;

    if !output.status.success() {
        return Err(format!(
            "gcc failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }

    Ok(())
}

fn fresh_test_dir() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("catena-executor-test-{nanos}"))
}

fn test_c_source() -> &'static str {
    r#"
    #include <stdint.h>

    void add_u64(uint64_t x, uint64_t y, uint64_t* out) {
        *out = x + y;
    }

    void mul_f32(float x, float y, float* out) {
        *out = x * y;
    }
    "#
}
