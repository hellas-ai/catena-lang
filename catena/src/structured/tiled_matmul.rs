use super::ir::{EntryPoint, Param, Primitive, Program, Stmt};
use super::ramsey::ArrowSemantics;
use std::collections::HashSet;

#[derive(Debug, Clone, Copy)]
pub struct TiledMatmulSemantics;

impl ArrowSemantics for TiledMatmulSemantics {
    fn actions(&self, op: &str) -> Vec<Stmt> {
        if self.counted_loop(op).is_some() {
            return Vec::new();
        }
        if op.contains(".barrier.") {
            return vec![Stmt::Barrier];
        }
        vec![Stmt::Primitive(primitive_mapping(op))]
    }

    fn counted_loop(&self, op: &str) -> Option<(String, String)> {
        if op.contains("for-num-tiles.prelude") {
            Some(("p".to_string(), "num_tiles".to_string()))
        } else if op.contains("for-tile.prelude") {
            Some(("q".to_string(), "TILE".to_string()))
        } else {
            None
        }
    }
}

pub fn program(definition: &str, body: Vec<Stmt>) -> Program {
    Program {
        name: sanitize_ident(definition),
        entry: EntryPoint {
            name: sanitize_ident(definition),
            params: vec![
                Param {
                    ty: "const float*".to_string(),
                    name: "A".to_string(),
                },
                Param {
                    ty: "const float*".to_string(),
                    name: "B".to_string(),
                },
                Param {
                    ty: "float*".to_string(),
                    name: "C".to_string(),
                },
                Param {
                    ty: "int".to_string(),
                    name: "M".to_string(),
                },
                Param {
                    ty: "int".to_string(),
                    name: "N".to_string(),
                },
                Param {
                    ty: "int".to_string(),
                    name: "K".to_string(),
                },
            ],
        },
        body,
    }
}

impl Program {
    pub fn render_c(&self) -> String {
        let branch_targets = BranchTargets::new(&self.body);
        let mut out = String::new();
        out.push_str("#include <stdint.h>\n\n");
        out.push_str("#ifndef TILE\n#define TILE 16\n#endif\n\n");
        out.push_str(&format!("__global__ void {}(", self.entry.name));
        out.push_str(
            &self
                .entry
                .params
                .iter()
                .map(|p| format!("{} {}", p.ty, p.name))
                .collect::<Vec<_>>()
                .join(", "),
        );
        out.push_str(") {\n");
        render_kernel_preamble(&mut out);
        render_c_stmts(&mut out, &self.body, 1, &branch_targets);
        out.push_str("}\n");
        out
    }
}

fn primitive_mapping(op: &str) -> Primitive {
    let code = match op {
        "gpu.tiled-matmul.init-acc" => "float acc = 0.0f;",
        "gpu.tiled-matmul.collectively-load-shared-tile" => {
            r#"int a_col = p * TILE + tx;
int b_row = p * TILE + ty;
tile_A[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
tile_B[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;"#
        }
        "gpu.tiled-matmul.dot-by-thread" => "acc += tile_A[ty][q] * tile_B[q][tx];",
        "gpu.tiled-matmul.store-output" => {
            r#"if (row < M && col < N) {
    C[row * N + col] = acc;
}"#
        }
        _ => {
            return Primitive {
                name: op.to_string(),
                code: format!("/* TODO: primitive {op}; */"),
            };
        }
    };

    Primitive {
        name: op.to_string(),
        code: code.to_string(),
    }
}

fn render_kernel_preamble(out: &mut String) {
    out.push_str("    __shared__ float tile_A[TILE][TILE];\n");
    out.push_str("    __shared__ float tile_B[TILE][TILE];\n\n");
    out.push_str("    int row = blockIdx.y * TILE + threadIdx.y;\n");
    out.push_str("    int col = blockIdx.x * TILE + threadIdx.x;\n");
    out.push_str("    int ty = threadIdx.y;\n");
    out.push_str("    int tx = threadIdx.x;\n");
    out.push_str("    int num_tiles = (K + TILE - 1) / TILE;\n\n");
}

fn render_c_stmts(out: &mut String, stmts: &[Stmt], indent: usize, branch_targets: &BranchTargets) {
    let pad = "    ".repeat(indent);
    for stmt in stmts {
        match stmt {
            Stmt::Block { label, body } => {
                out.push_str(&format!("{pad}do {{\n"));
                render_c_stmts(out, body, indent + 1, branch_targets);
                out.push_str(&format!("{pad}}} while (0);\n"));
                if branch_targets.breaks.contains(label) {
                    out.push_str(&format!("{pad}{}:\n", after_label(label)));
                }
            }
            Stmt::Loop { label, body } => {
                out.push_str(&format!("{pad}while (1) {{\n"));
                if branch_targets.continues.contains(label) {
                    out.push_str(&format!("{pad}{}:\n", continue_label(label)));
                }
                render_c_stmts(out, body, indent + 1, branch_targets);
                out.push_str(&format!("{pad}}}\n"));
                if branch_targets.breaks.contains(label) {
                    out.push_str(&format!("{pad}{}:\n", after_label(label)));
                }
            }
            Stmt::For {
                label,
                var,
                extent,
                body,
            } => {
                out.push_str(&format!(
                    "{pad}for (int {var} = 0; {var} < {extent}; ++{var}) {{\n"
                ));
                render_c_stmts(out, body, indent + 1, branch_targets);
                if branch_targets.continues.contains(label) {
                    out.push_str(&format!("{pad}{}:\n", continue_label(label)));
                }
                out.push_str(&format!("{pad}}}\n"));
                if branch_targets.breaks.contains(label) {
                    out.push_str(&format!("{pad}{}:\n", after_label(label)));
                }
            }
            Stmt::If {
                condition,
                then_body,
                else_body,
            } => {
                out.push_str(&format!("{pad}if ({condition}) {{\n"));
                render_c_stmts(out, then_body, indent + 1, branch_targets);
                out.push_str(&format!("{pad}}} else {{\n"));
                render_c_stmts(out, else_body, indent + 1, branch_targets);
                out.push_str(&format!("{pad}}}\n"));
            }
            Stmt::Switch { selector, cases } => {
                out.push_str(&format!("{pad}switch ({selector}) {{\n"));
                for (index, body) in cases.iter().enumerate() {
                    out.push_str(&format!("{pad}case {index}:\n"));
                    render_c_stmts(out, body, indent + 1, branch_targets);
                    out.push_str(&format!("{pad}    break;\n"));
                }
                out.push_str(&format!("{pad}}}\n"));
            }
            Stmt::Break(label) => out.push_str(&format!("{pad}goto {};\n", after_label(label))),
            Stmt::Continue(label) => {
                out.push_str(&format!("{pad}goto {};\n", continue_label(label)))
            }
            Stmt::Return => out.push_str(&format!("{pad}return;\n")),
            Stmt::Barrier => out.push_str(&format!("{pad}__syncthreads();\n")),
            Stmt::Primitive(primitive) => {
                for line in primitive.code.lines() {
                    out.push_str(&format!("{pad}{line}\n"));
                }
            }
            Stmt::Comment(comment) => out.push_str(&format!("{pad}// {comment}\n")),
        }
    }
}

#[derive(Debug, Default)]
struct BranchTargets {
    breaks: HashSet<String>,
    continues: HashSet<String>,
}

impl BranchTargets {
    fn new(stmts: &[Stmt]) -> Self {
        let mut targets = Self::default();
        targets.collect(stmts);
        targets
    }

    fn collect(&mut self, stmts: &[Stmt]) {
        for stmt in stmts {
            match stmt {
                Stmt::Block { body, .. } | Stmt::Loop { body, .. } | Stmt::For { body, .. } => {
                    self.collect(body);
                }
                Stmt::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    self.collect(then_body);
                    self.collect(else_body);
                }
                Stmt::Switch { cases, .. } => {
                    for body in cases {
                        self.collect(body);
                    }
                }
                Stmt::Break(label) => {
                    self.breaks.insert(label.clone());
                }
                Stmt::Continue(label) => {
                    self.continues.insert(label.clone());
                }
                Stmt::Return | Stmt::Barrier | Stmt::Primitive(_) | Stmt::Comment(_) => {}
            }
        }
    }
}

fn after_label(label: &str) -> String {
    format!("{label}_after")
}

fn continue_label(label: &str) -> String {
    format!("{label}_continue")
}

fn sanitize_ident(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}
