//! Run with --features runtime-tests --test file_memory_gpu on a GPU host.
//! Set TMPDIR to the filesystem whose mapping behavior is under test.
use catena_lang::runtime::{
    ExecError, GeneratedFunction, GpuDialect, MemOwn, Runtime, RuntimeModule, ValueKind,
};
use catena_lang::safe_runtime::{SafeExecError, SafeRuntime, run_safe_runtime_child_if_requested};
use std::{env, fs::File, io::Write};

fn module(dialect: GpuDialect) -> RuntimeModule {
    let api = match dialect {
        GpuDialect::Hip => "hip",
        GpuDialect::Cuda => "cuda",
    };
    let header = if dialect == GpuDialect::Hip {
        "hip/hip_runtime.h"
    } else {
        "cuda_runtime.h"
    };
    let source = format!(
        r#"
#include <{header}>
#include <cstdint>
#include <cstdlib>
struct Mem {{ void* data; uint64_t len; }};
__global__ void sum(Mem a, Mem b, uint64_t* out) {{
  uint64_t total = 0;
  for (uint64_t i=0; i<a.len; ++i) total += ((const unsigned char*)a.data)[i];
  for (uint64_t i=0; i<b.len; ++i) total += ((const unsigned char*)b.data)[i];
  *out = total;
}}
extern "C" void read_file(Mem a, Mem b, uint64_t* out) {{
  uint64_t* p = nullptr;
  if ({api}Malloc(&p, sizeof(uint64_t))) abort();
  sum<<<1,1>>>(a,b,p);
  if ({api}Memcpy(out,p,sizeof(uint64_t),{api}MemcpyDeviceToHost)) abort();
  if ({api}Free(p)) abort();
}}
extern "C" void consume(Mem a) {{ if ({api}Free(a.data)) abort(); }}
"#
    );
    RuntimeModule::new(
        dialect,
        source,
        [
            GeneratedFunction {
                source_name: "read_file".into(),
                symbol: "read_file".into(),
                inputs: vec![ValueKind::MemRef; 2],
                outputs: vec![ValueKind::U64],
            },
            GeneratedFunction {
                source_name: "consume".into(),
                symbol: "consume".into(),
                inputs: vec![ValueKind::MemOwn],
                outputs: vec![],
            },
        ],
    )
}

fn load(file: File, dialect: GpuDialect) -> Result<MemOwn, Box<dyn std::error::Error>> {
    // This test never modifies the file or writes through its device views.
    Ok(unsafe { MemOwn::from_file(file, dialect)? })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if run_safe_runtime_child_if_requested()? {
        return Ok(());
    }
    let dialect = if env::var("CATENA_TEST_DIALECT").as_deref() == Ok("cuda") {
        GpuDialect::Cuda
    } else {
        GpuDialect::Hip
    };
    let mut file = tempfile::NamedTempFile::new()?;
    let bytes: Vec<u8> = (0..2 * 1024 * 1024 + 17).map(|i| (i % 251) as u8).collect();
    file.write_all(&bytes)?;
    file.flush()?;
    let descriptor = file.reopen()?;
    // Exercise procfs descriptor transport after the original pathname vanishes.
    file.close()?;
    let memory = load(descriptor.try_clone()?, dialect)?;
    assert_eq!(memory.byte_len(), bytes.len() as u64);
    let expected = bytes[7..1034]
        .iter()
        .chain(&bytes[129..4226])
        .map(|&v| u64::from(v))
        .sum::<u64>();
    let mut runtime = Runtime::new(dialect)?;
    let artifact = runtime.load(module(dialect))?;
    let args = || {
        vec![
            memory.view(7, 1027).unwrap().into(),
            memory.view(129, 4097).unwrap().into(),
        ]
    };
    let result = artifact.exec_values("read_file", args())?;
    assert!(matches!(&result[..], [catena_lang::runtime::Value::U64(value)] if *value == expected));
    let mut safe = SafeRuntime::new(dialect)?;
    let remote = safe.load(module(dialect))?;
    for _ in 0..2 {
        let result = remote.exec_values("read_file", args())?;
        assert!(
            matches!(&result[..], [catena_lang::runtime::Value::U64(value)] if *value == expected)
        );
    }
    // A successful read on an integrated device must not permit ownership transfer.
    // Discrete devices return normal allocations and may transfer them.
    let owned = load(descriptor.try_clone()?, dialect)?;
    let local_owned = artifact.exec_values("consume", vec![owned.into()]);
    let owned = load(descriptor, dialect)?;
    let remote_owned = remote.exec_values("consume", vec![owned.into()]);
    match (local_owned, remote_owned) {
        (
            Err(ExecError::FileRequiresBorrow { index: 0 }),
            Err(SafeExecError::Runtime(ExecError::FileRequiresBorrow { index: 0 })),
        ) => {}
        (Ok(_), Ok(_)) => {}
        (local, remote) => panic!("inconsistent owned-file behavior: {local:?}; {remote:?}"),
    }
    println!(
        "file memory GPU: unlinked file, ranged reads, Runtime and SafeRuntime passed; checksum={expected}"
    );
    Ok(())
}
