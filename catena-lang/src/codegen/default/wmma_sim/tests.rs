//! Numerical validation only: the harness executes emitted production helpers.
use super::super::{
    Lowerer, expr, gpu,
    ir::*,
    lower_types::{CType, node},
    values::{Repr, Value},
};
use crate::runtime::GpuDialect;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, process::Command};

fn variable(name: &str, ty: CType) -> Variable {
    Variable {
        name: name.into(),
        ty,
    }
}
fn let_value(name: &str, ty: CType, expression: &str) -> Instruction {
    Instruction::Let(variable(name, ty), expression.into())
}
fn assign(target: &str, expression: &str) -> Instruction {
    Instruction::Assign(target.into(), expression.into())
}
fn loop8(body: Vec<Instruction>) -> Instruction {
    Instruction::For {
        index: "i".into(),
        index_type: CType::U32,
        end: "8".into(),
        body,
    }
}

// Use the real primitive lowerer so numerical tests cover its call signature too.
fn lower_primitive(
    lower: &mut Lowerer<'_>,
    op: &str,
    inputs: &[(&str, &str)],
    output: &str,
) -> (Vec<Instruction>, String) {
    let args = inputs
        .iter()
        .map(|(ty, expression)| Value {
            ty: node("val", vec![node(ty, vec![])]),
            repr: Repr::Runtime((*expression).into()),
        })
        .collect();
    let result = lower
        .lower_operation(op, args, &[node("val", vec![node(output, vec![])])])
        .unwrap();
    let expression = expr(&result[0]).unwrap().to_owned();
    (std::mem::take(&mut lower.body), expression)
}

/// Render a real launch with the production hidden argument and environment read.
/// Four independent tiles per block exercise subgroup isolation and partial batches.
fn generated_module(dialect: GpuDialect) -> String {
    let templates = Default::default();
    let mut lower = Lowerer::new(&templates);
    lower.place = Place::Device;
    let inputs = || {
        vec![
            variable("a", CType::MemRef),
            variable("b", CType::MemRef),
            variable("c", CType::MemRef),
            variable("d", CType::MemRef),
            variable("count", CType::U64),
            variable("repetitions", CType::U64),
            variable("shape", CType::U64),
        ]
    };
    let (mut work, zero) = lower_primitive(&mut lower, "wmma.f32x8.zero", &[], "wmma.f32x8");
    work.extend(vec![
        let_value("column", CType::U32, "linear % 16"),
        let_value("parity", CType::U32, "(linear % 32) / 16"),
        let_value("af", CType::WmmaBf16x16, "{}"),
        let_value("bf", CType::WmmaBf16x16, "{}"),
        let_value("acc", CType::WmmaF32x8, &zero),
        loop8(vec![
            assign(
                "af.pair[i]",
                "catena_bf16_load_pair(static_cast<const uint16_t*>(a.data) + tile*256 + column*16 + 2*i).bits",
            ),
            assign(
                "bf.pair[i]",
                "catena_bf16_load_pair(static_cast<const uint16_t*>(b.data) + tile*256 + column*16 + 2*i).bits",
            ),
            assign(
                "acc.value[i]",
                "catena_u32_bitcast_f32(static_cast<const uint32_t*>(c.data)[tile*256 + (parity+2*i)*16 + column])",
            ),
        ]),
    ]);
    for fragment in ["af", "bf"] {
        let pairs: Vec<_> = (0..8)
            .map(|i| format!("catena_bf16x2_t{{{fragment}.pair[{i}]}}"))
            .collect();
        let args: Vec<_> = pairs.iter().map(|p| ("bf16x2", p.as_str())).collect();
        let (pack, result) =
            lower_primitive(&mut lower, "wmma.bf16x16.pack", &args, "wmma.bf16x16");
        work.extend(pack);
        work.push(assign(fragment, &result));
    }
    let values: Vec<_> = (0..8).map(|i| format!("acc.value[{i}]")).collect();
    let args: Vec<_> = values.iter().map(|v| ("f32", v.as_str())).collect();
    let (pack, result) = lower_primitive(&mut lower, "wmma.f32x8.pack", &args, "wmma.f32x8");
    work.extend(pack);
    work.push(assign("acc", &result));
    let (mut step, result) = lower_primitive(
        &mut lower,
        "wmma.mma-bf16-f32",
        &[
            ("wmma.bf16x16", "af"),
            ("wmma.bf16x16", "bf"),
            ("wmma.f32x8", "acc"),
        ],
        "wmma.f32x8",
    );
    step.push(assign("acc", &result));
    work.push(Instruction::For {
        index: "repeat".into(),
        index_type: CType::U64,
        end: "repetitions".into(),
        body: step,
    });
    let (mut store, result) = lower_primitive(
        &mut lower,
        "wmma.f32x8.get",
        &[("wmma.f32x8", "acc"), ("u32", "i")],
        "f32",
    );
    store.push(assign(
        "static_cast<uint32_t*>(d.data)[tile*256 + (parity+2*i)*16 + column]",
        &format!("catena_f32_bitcast_u32({result})"),
    ));
    work.push(loop8(store));
    let kernel = Function {
        symbol: "catena_stress_kernel".into(),
        inputs: inputs(),
        outputs: vec![],
        body: vec![
            let_value(
                "linear",
                CType::U32,
                "threadIdx.x + blockDim.x*(threadIdx.y + blockDim.y*threadIdx.z)",
            ),
            let_value("tile", CType::U64, "uint64_t(blockIdx.x)*4 + linear/32"),
            Instruction::If {
                condition: "tile < count".into(),
                yes: work,
                no: vec![],
            },
        ],
    };
    let host = Function {
        symbol: "catena_stress_launch".into(),
        inputs: inputs(),
        outputs: vec![],
        body: vec![
            let_value(
                "grid",
                CType::Grid,
                "{dim3((count+3)/4), shape == 0 ? dim3(128) : shape == 1 ? dim3(16,8) : dim3(8,4,4)}",
            ),
            Instruction::Launch {
                kernel: kernel.symbol.clone(),
                grid: "grid".into(),
                shared_bytes: "0".into(),
                arguments: inputs().iter().map(|v| v.name.clone()).collect(),
            },
        ],
    };
    let mut modules = Modules {
        bf16: true,
        wmma: true,
        ..Default::default()
    };
    modules.kernels.insert(kernel.symbol.clone(), kernel);
    modules.functions.insert(host.symbol.clone(), host);
    gpu::render(&modules, dialect)
}

fn harness(dialect: Option<GpuDialect>) -> String {
    let mut source = if let Some(dialect) = dialect {
        format!(
            "{}\n{}",
            include_str!("gpu_test.h"),
            generated_module(dialect)
        )
    } else {
        format!(
            "#define WMMA_CPU_ONLY\n#define __host__\n#define __device__\n#include <stdint.h>\n{}",
            include_str!("arithmetic.h")
        )
    };
    source.push_str(
        &include_str!("stress.cpp")
            .replace("#include \"reference.h\"", include_str!("reference.h"))
            .replace("#include \"corpus.h\"", include_str!("corpus.h"))
            .replace("#include \"reference64.h\"", include_str!("reference64.h"))
            .replace("#include \"suites.h\"", include_str!("suites.h")),
    );
    let hash = format!("{:x}", Sha256::digest(source.as_bytes()));
    format!("#define WMMA_SOURCE_SHA256 \"{hash}\"\n{source}")
}

fn run_stress(dialect: Option<GpuDialect>) {
    let temporary = tempfile::tempdir().unwrap();
    let directory = std::env::var_os("CATENA_WMMA_TEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| temporary.path().to_owned());
    fs::create_dir_all(&directory).unwrap();
    let directory = fs::canonicalize(directory).unwrap();
    let name = match dialect {
        None => "cpu",
        Some(GpuDialect::Hip) => "hip",
        Some(GpuDialect::Cuda) => "cuda",
    };
    let source = directory.join(format!("wmma-stress-{name}.cpp"));
    let binary = directory.join(format!("wmma-stress-{name}"));
    fs::write(&source, harness(dialect)).unwrap();
    let compiler = match dialect {
        None => std::env::var_os("CXX").unwrap_or_else(|| "c++".into()),
        Some(GpuDialect::Hip) => std::env::var_os("HIPCC").unwrap_or_else(|| "hipcc".into()),
        Some(GpuDialect::Cuda) => std::env::var_os("NVCC").unwrap_or_else(|| "nvcc".into()),
    };
    let mut build = Command::new(compiler);
    build.args(["-O2", "-std=c++17"]);
    match dialect {
        Some(GpuDialect::Hip) => {
            let arch = std::env::var("CATENA_WMMA_TEST_ARCH").unwrap_or_else(|_| "gfx1151".into());
            build.arg(format!("--offload-arch={arch}")).args([
                "-mno-wavefrontsize64",
                "-pthread",
                "-fno-fast-math",
                "-ffp-contract=off",
            ]);
        }
        Some(GpuDialect::Cuda) => {
            let arch = std::env::var("CATENA_WMMA_TEST_ARCH").unwrap_or_else(|_| "sm_80".into());
            build
                .args([
                    "-x",
                    "cu",
                    "--fmad=false",
                    "-Xcompiler=-pthread,-fno-fast-math,-ffp-contract=off",
                ])
                .arg(format!("-arch={arch}"));
        }
        None => {
            build.args(["-pthread", "-fno-fast-math", "-ffp-contract=off"]);
        }
    }
    let built = build
        .arg(&source)
        .arg("-o")
        .arg(&binary)
        .output()
        .expect("start stress harness compiler");
    if !built.status.success() {
        let _ = temporary.keep();
        panic!(
            "stress harness compilation failed in {}:\n{}",
            directory.display(),
            String::from_utf8_lossy(&built.stderr)
        );
    }
    // Build-only is useful on a development host with an SDK but no GPU.
    if dialect.is_some() && std::env::var_os("CATENA_WMMA_TEST_BUILD_ONLY").is_some() {
        eprintln!("Compiled {}; GPU execution NOT checked", binary.display());
        return;
    }
    let mut run = Command::new(&binary);
    run.current_dir(&directory);
    for (variable, argument) in [
        ("CATENA_WMMA_TEST_TRIALS", "--trials"),
        ("CATENA_WMMA_TEST_SEED", "--seed"),
        ("CATENA_WMMA_TEST_BATCH", "--batch"),
        ("CATENA_WMMA_TEST_REPLAY", "--replay"),
    ] {
        if let Some(value) = std::env::var_os(variable) {
            run.arg(argument).arg(value);
        }
    }
    run.arg("--report")
        .arg(directory.join(format!("wmma-stress-{name}.json")));
    let output = run.output().expect("run stress harness");
    eprintln!("{}", String::from_utf8_lossy(&output.stdout));
    if !output.status.success() {
        let _ = temporary.keep();
        panic!(
            "WMMA stress validation failed in {}:\n{}",
            directory.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn cpu_stress() {
    run_stress(None);
}

#[cfg(feature = "runtime-tests")]
#[test]
#[ignore = "requires RDNA3/3.5 GPU and HIP SDK; see wmma_sim/README.md"]
fn hip_stress() {
    run_stress(Some(GpuDialect::Hip));
}

#[cfg(feature = "runtime-tests")]
#[test]
#[ignore = "requires NVIDIA GPU and CUDA SDK; see wmma_sim/README.md"]
fn cuda_stress() {
    run_stress(Some(GpuDialect::Cuda));
}

#[test]
fn wmma_rejects_host_execution() {
    let templates = Default::default();
    let mut lower = Lowerer::new(&templates);
    let result = lower.lower_operation(
        "wmma.f32x8.zero",
        vec![],
        &[node("val", vec![node("wmma.f32x8", vec![])])],
    );
    assert!(
        matches!(result, Err(super::super::CodegenError::Invalid { reason, .. }) if reason == "WMMA requires device execution")
    );
}
