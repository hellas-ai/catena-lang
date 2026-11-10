//! Included only for reachable NVIDIA fragment operations/types. build.rs hashes
//! every file under src, including this helper and its C++ source.
pub(super) const SUPPORT: &str = include_str!("nvidia_mma.h");

#[cfg(test)]
mod tests {
    use super::super::lower_types::CType;
    use super::super::*;

    fn scalar(name: &str) -> Obj {
        node("val", vec![node(name, vec![])])
    }
    fn value(kind: &str, name: &str) -> Value {
        Value {
            ty: scalar(kind),
            repr: Repr::Runtime(name.into()),
        }
    }
    const A: &str = "mma.m16n8k16.bf16.a";
    const B: &str = "mma.m16n8k16.bf16.b";
    const C: &str = "mma.m16n8k16.f32.acc";

    #[test]
    fn operations_check_every_operand_result_and_place() {
        let templates = BTreeMap::new();
        for (op, inputs, output) in [
            ("mma.m16n8k16.bf16.a.pack", vec!["bf16x2"; 4], A),
            ("mma.m16n8k16.bf16.b.pack", vec!["bf16x2"; 2], B),
            ("mma.m16n8k16.f32.acc.pack", vec!["f32"; 4], C),
            ("mma.m16n8k16.f32.acc.zero", vec![], C),
            ("mma.m16n8k16.f32.acc.get", vec![C, "u32"], "f32"),
            ("mma.m16n8k16.bf16-f32", vec![A, B, C], C),
        ] {
            let args = inputs
                .iter()
                .enumerate()
                .map(|(i, ty)| value(ty, &format!("x{i}")))
                .collect::<Vec<_>>();
            let mut l = Lowerer::new(&templates);
            assert!(
                l.lower_operation(op, args.clone(), &[scalar(output)])
                    .is_err()
            );
            l.place = Place::Device;
            for i in 0..args.len() {
                let mut bad = args.clone();
                bad[i] = value("u64", "bad");
                assert!(l.lower_operation(op, bad, &[scalar(output)]).is_err());
            }
            let mut extra = args.clone();
            extra.push(value("u64", "extra"));
            assert!(l.lower_operation(op, extra, &[scalar(output)]).is_err());
            assert!(
                l.lower_operation(op, args.clone(), &[scalar("bool")])
                    .is_err()
            );
            assert!(l.body.is_empty());
            l.lower_operation(op, args, &[scalar(output)]).unwrap();
            assert!(l.modules.nvidia_mma && l.modules.bf16);
            assert_eq!(l.body.len(), 1);
        }
        assert_eq!(runtime(&scalar(A)).unwrap(), Some(CType::MmaBf16A));
        assert_eq!(runtime(&scalar(B)).unwrap(), Some(CType::MmaBf16B));
        assert_eq!(runtime(&scalar(C)).unwrap(), Some(CType::MmaF32Acc));
    }

    fn module() -> Modules {
        let templates = BTreeMap::new();
        let mut l = Lowerer::new(&templates);
        l.place = Place::Device;
        let result = l
            .lower_operation(
                "mma.m16n8k16.bf16-f32",
                vec![value(A, "a"), value(B, "b"), value(C, "acc")],
                &[scalar(C)],
            )
            .unwrap();
        assert!(
            matches!(&l.body[0], Instruction::NvidiaMma { result: r, a, b, accumulator }
            if r.name == expr(&result[0]).unwrap() && a.name == "a" && b.name == "b" && accumulator.name == "acc")
        );
        l.modules.kernels.insert(
            "kernel".into(),
            Function {
                symbol: "kernel".into(),
                inputs: [(A, "a"), (B, "b"), (C, "acc")]
                    .into_iter()
                    .map(|(t, n)| Variable {
                        name: n.into(),
                        ty: runtime(&scalar(t)).unwrap().unwrap(),
                    })
                    .collect(),
                outputs: vec![],
                body: vec![Instruction::If {
                    condition: "true".into(),
                    yes: l.body,
                    no: vec![],
                }],
            },
        );
        l.modules
    }

    #[test]
    fn typed_ir_rejects_swapped_layouts_missing_support_host_and_hip() {
        let modules = module();
        assert!(render_runtime_module(&modules, GpuDialect::Cuda).is_ok());
        let error = render_runtime_module(&modules, GpuDialect::Hip)
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires CUDA"));
        for index in 0..4 {
            let mut bad = modules.clone();
            let Instruction::If { yes, .. } = &mut bad.kernels.get_mut("kernel").unwrap().body[0]
            else {
                panic!()
            };
            let Instruction::NvidiaMma {
                result,
                a,
                b,
                accumulator,
            } = &mut yes[0]
            else {
                panic!()
            };
            [result, a, b, accumulator][index].ty = CType::F32;
            assert!(validate::validate(&bad).is_err());
        }
        let mut bad = modules.clone();
        let Instruction::If { yes, .. } = &mut bad.kernels.get_mut("kernel").unwrap().body[0]
        else {
            panic!()
        };
        let Instruction::NvidiaMma { a, b, .. } = &mut yes[0] else {
            panic!()
        };
        std::mem::swap(&mut a.ty, &mut b.ty);
        assert!(validate::validate(&bad).is_err());
        for flag in ["bf16", "mma"] {
            let mut bad = modules.clone();
            if flag == "bf16" {
                bad.bf16 = false;
            } else {
                bad.nvidia_mma = false;
            }
            assert!(validate::validate(&bad).is_err());
        }
        let mut bad = modules;
        bad.functions = std::mem::take(&mut bad.kernels);
        assert!(validate::validate(&bad).is_err());
    }

    #[test]
    fn source_preserves_register_lists_constraints_and_bounds() {
        let source = render_runtime_module(&module(), GpuDialect::Cuda)
            .unwrap()
            .source;
        assert!(source.contains("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"));
        assert!(source.contains("{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"));
        assert!(source.contains("catena_mma_bf16_f32(a,b,acc)"));
        assert!(source.contains("catena_assert(i < 4); return a.elt[i];"));
        assert!(source.contains("return {{a.bits,b.bits,c.bits,d.bits}};"));
        assert!(source.contains("return {{a.bits,b.bits}};"));
        assert_eq!(source.matches("\"=f\"(d.elt[").count(), 4);
        assert_eq!(source.matches("\"r\"(a.pair[").count(), 4);
        assert_eq!(source.matches("\"r\"(b.pair[").count(), 2);
        assert_eq!(source.matches("\"f\"(c.elt[").count(), 4);
        assert!(source.contains("__CUDA_ARCH__ >= 800"));
        assert!(!source.contains("catena_wmma"));
        for dialect in [GpuDialect::Cuda, GpuDialect::Hip] {
            let source = render_runtime_module(&Modules::default(), dialect)
                .unwrap()
                .source;
            assert!(!source.contains("catena_mma"));
        }
    }

    #[test]
    #[ignore = "requires NVIDIA CUDA toolkit (nvcc), no GPU required"]
    fn cuda_toolkit_compiles_generated_source_and_rejects_sm75() {
        use std::{fs, process::Command};
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("mma.cu");
        fs::write(
            &source,
            render_runtime_module(&module(), GpuDialect::Cuda)
                .unwrap()
                .source,
        )
        .unwrap();
        for fmad in ["true", "false"] {
            let ptx = dir.path().join(format!("mma-{fmad}.ptx"));
            let output = Command::new("nvcc")
                .args([
                    "-std=c++17",
                    "-arch=sm_80",
                    "--ptx",
                    &format!("--fmad={fmad}"),
                ])
                .arg(&source)
                .arg("-o")
                .arg(&ptx)
                .output()
                .expect("requires nvcc");
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(
                fs::read_to_string(ptx)
                    .unwrap()
                    .contains("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32")
            );
        }
        let output = Command::new("nvcc")
            .args(["-std=c++17", "-arch=sm_75", "--ptx"])
            .arg(source)
            .arg("-o")
            .arg(dir.path().join("unsupported.ptx"))
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains("NVIDIA BF16 m16n8k16 MMA requires sm_80 or later")
        );
    }
    #[test]
    fn fragment_packing_and_checked_access_execute_on_cpu() {
        use std::{fs, process::Command};
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("fragments.cpp");
        let text = format!(
            r#"
#include <cstdint>
#include <stdexcept>
#define __device__
struct catena_bf16x2_t {{ uint32_t bits; }};
inline void catena_assert(bool ok) {{ if (!ok) throw std::out_of_range("index"); }}
{}
int main() {{
    auto a=catena_mma_a_pack({{0x01234567}},{{0x89abcdef}},{{0xfedcba98}},{{0x76543210}});
    auto b=catena_mma_b_pack({{0x12345678}},{{0x9abcdef0}});
    if(a.pair[0]!=0x01234567 || a.pair[1]!=0x89abcdef || a.pair[2]!=0xfedcba98 || a.pair[3]!=0x76543210) return 1;
    if(b.pair[0]!=0x12345678 || b.pair[1]!=0x9abcdef0) return 2;
    catena_mma_f32_acc_t c{{{{1.0f,-2.0f,3.5f,-4.25f}}}}, z{{{{0,0,0,0}}}};
    for(unsigned i=0;i<4;++i) if(catena_mma_acc_get(c,i)!=c.elt[i] || catena_mma_acc_get(z,i)!=0) return 3;
    for(unsigned i : {{4u,0xffffffffu}}) {{
        try {{ catena_mma_acc_get(c,i); return 4; }} catch(const std::out_of_range&) {{}}
    }}
}}
"#,
            super::SUPPORT
        );
        fs::write(&source, text).unwrap();
        let binary = dir.path().join("fragments");
        let output = Command::new("c++")
            .arg("-std=c++17")
            .arg(source)
            .arg("-o")
            .arg(&binary)
            .output()
            .expect("requires a C++ compiler");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(Command::new(binary).status().unwrap().success());
    }

    #[test]
    #[ignore = "requires nvcc and an NVIDIA sm_80+ GPU; set CATENA_MMA_BUILD_ONLY=1 for toolkit-only compilation"]
    fn cuda_hardware_fragments_and_products() {
        use std::{fs, process::Command};
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("hardware.cu");
        let mut text = render_runtime_module(
            &Modules {
                bf16: true,
                nvidia_mma: true,
                ..Default::default()
            },
            GpuDialect::Cuda,
        )
        .unwrap()
        .source;
        text.push_str(include_str!("nvidia_mma_test.cu"));
        fs::write(&source, text).unwrap();
        let binary = dir.path().join("hardware");
        let output = Command::new("nvcc")
            .args(["-std=c++17", "-arch=sm_80", "--fmad=false", "-Xptxas=-v"])
            .arg(source)
            .arg("-o")
            .arg(&binary)
            .output()
            .expect("requires nvcc");
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        assert!(output.status.success());
        let sass = Command::new("cuobjdump")
            .arg("--dump-sass")
            .arg(&binary)
            .output()
            .expect("requires CUDA toolkit cuobjdump");
        assert!(
            sass.status.success(),
            "{}",
            String::from_utf8_lossy(&sass.stderr)
        );
        assert!(
            String::from_utf8_lossy(&sass.stdout).contains("HMMA"),
            "expected native tensor-core instructions"
        );
        if std::env::var("CATENA_MMA_BUILD_ONLY").as_deref() != Ok("1") {
            assert!(
                Command::new(binary)
                    .status()
                    .expect("requires NVIDIA sm_80+ GPU")
                    .success()
            );
        }
    }
}
