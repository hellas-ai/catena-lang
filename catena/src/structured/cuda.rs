use super::ir::{Param, Primitive, Stmt, StructuredProgram};
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaRenderMode {
    Kernel,
    KernelWithLaunch,
}

#[derive(Debug, Clone)]
pub struct CudaKernelEnv {
    pub tile_macro: String,
    pub tile_size: usize,
    pub params: Vec<Param>,
    pub shared: Vec<CudaDecl>,
    pub prelude: Vec<CudaStmt>,
    pub launch: Option<CudaLaunchConfig>,
}

#[derive(Debug, Clone)]
pub struct CudaLaunchConfig {
    pub block: String,
    pub grid: String,
}

#[derive(Debug, Clone)]
pub struct CudaDecl {
    pub ty: String,
    pub name: String,
    pub init: Option<String>,
}

#[derive(Debug, Clone)]
pub enum CudaStmt {
    Decl(CudaDecl),
    Assign {
        lhs: String,
        rhs: String,
    },
    AddAssign {
        lhs: String,
        rhs: String,
    },
    If {
        condition: String,
        body: Vec<CudaStmt>,
    },
    Syncthreads,
}

#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("no CUDA lowering for primitive {0}")]
    UnknownPrimitive(String),
}

pub fn render_cuda(
    program: &StructuredProgram,
    env: &CudaKernelEnv,
    mode: CudaRenderMode,
    lower_primitive: impl Fn(&Primitive) -> Result<Vec<CudaStmt>, CudaError>,
) -> Result<String, CudaError> {
    let branch_targets = BranchTargets::new(&program.body);
    let mut out = String::new();
    out.push_str("#include <stdint.h>\n\n");
    out.push_str(&format!(
        "#ifndef {tile}\n#define {tile} {tile_size}\n#endif\n\n",
        tile = env.tile_macro,
        tile_size = env.tile_size
    ));
    out.push_str(&format!("__global__ void {}(", program.entry.name));
    out.push_str(
        &program
            .entry
            .params
            .iter()
            .map(|p| format!("{} {}", p.ty, p.name))
            .collect::<Vec<_>>()
            .join(", "),
    );
    out.push_str(") {\n");
    render_kernel_preamble(&mut out, env);
    render_cuda_stmts(
        &mut out,
        &program.body,
        1,
        &branch_targets,
        &lower_primitive,
    )?;
    out.push_str("}\n");

    if mode == CudaRenderMode::KernelWithLaunch {
        if let Some(launch) = &env.launch {
            out.push('\n');
            render_launch_helper(&mut out, program, launch);
        }
    }

    Ok(out)
}

fn render_kernel_preamble(out: &mut String, env: &CudaKernelEnv) {
    for decl in &env.shared {
        out.push_str(&format!("    __shared__ {} {};\n", decl.ty, decl.name));
    }
    out.push('\n');
    render_cuda_stmt_list(out, &env.prelude, 1);
    out.push('\n');
}

fn render_launch_helper(out: &mut String, program: &StructuredProgram, launch: &CudaLaunchConfig) {
    out.push_str(&format!("void launch_{}(", program.entry.name));
    out.push_str(
        &program
            .entry
            .params
            .iter()
            .map(|p| format!("{} {}", p.ty, p.name))
            .collect::<Vec<_>>()
            .join(", "),
    );
    out.push_str(") {\n");
    out.push_str(&format!("    dim3 block({});\n", launch.block));
    out.push_str(&format!("    dim3 grid({});\n", launch.grid));
    out.push_str(&format!(
        "    {}<<<grid, block>>>({});\n",
        program.entry.name,
        program
            .entry
            .params
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    ));
    out.push_str("}\n");
}

fn render_cuda_stmts(
    out: &mut String,
    stmts: &[Stmt],
    indent: usize,
    branch_targets: &BranchTargets,
    lower_primitive: &impl Fn(&Primitive) -> Result<Vec<CudaStmt>, CudaError>,
) -> Result<(), CudaError> {
    let pad = "    ".repeat(indent);
    for stmt in stmts {
        match stmt {
            Stmt::Block { label, body } => {
                out.push_str(&format!("{pad}do {{\n"));
                render_cuda_stmts(out, body, indent + 1, branch_targets, lower_primitive)?;
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
                render_cuda_stmts(out, body, indent + 1, branch_targets, lower_primitive)?;
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
                render_cuda_stmts(out, body, indent + 1, branch_targets, lower_primitive)?;
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
                render_cuda_stmts(out, then_body, indent + 1, branch_targets, lower_primitive)?;
                out.push_str(&format!("{pad}}} else {{\n"));
                render_cuda_stmts(out, else_body, indent + 1, branch_targets, lower_primitive)?;
                out.push_str(&format!("{pad}}}\n"));
            }
            Stmt::Switch { selector, cases } => {
                out.push_str(&format!("{pad}switch ({selector}) {{\n"));
                for (index, body) in cases.iter().enumerate() {
                    out.push_str(&format!("{pad}case {index}:\n"));
                    render_cuda_stmts(out, body, indent + 1, branch_targets, lower_primitive)?;
                    out.push_str(&format!("{pad}    break;\n"));
                }
                out.push_str(&format!("{pad}}}\n"));
            }
            Stmt::Break(label) => out.push_str(&format!("{pad}goto {};\n", after_label(label))),
            Stmt::Continue(label) => {
                out.push_str(&format!("{pad}goto {};\n", continue_label(label)))
            }
            Stmt::Return => out.push_str(&format!("{pad}return;\n")),
            Stmt::Barrier => render_cuda_stmt(out, &CudaStmt::Syncthreads, indent),
            Stmt::Primitive(primitive) => {
                render_cuda_stmt_list(out, &lower_primitive(primitive)?, indent);
            }
            Stmt::Comment(comment) => out.push_str(&format!("{pad}// {comment}\n")),
        }
    }
    Ok(())
}

fn render_cuda_stmt_list(out: &mut String, stmts: &[CudaStmt], indent: usize) {
    for stmt in stmts {
        render_cuda_stmt(out, stmt, indent);
    }
}

fn render_cuda_stmt(out: &mut String, stmt: &CudaStmt, indent: usize) {
    let pad = "    ".repeat(indent);
    match stmt {
        CudaStmt::Decl(decl) => match &decl.init {
            Some(init) => out.push_str(&format!("{pad}{} {} = {};\n", decl.ty, decl.name, init)),
            None => out.push_str(&format!("{pad}{} {};\n", decl.ty, decl.name)),
        },
        CudaStmt::Assign { lhs, rhs } => out.push_str(&format!("{pad}{lhs} = {rhs};\n")),
        CudaStmt::AddAssign { lhs, rhs } => out.push_str(&format!("{pad}{lhs} += {rhs};\n")),
        CudaStmt::If { condition, body } => {
            out.push_str(&format!("{pad}if ({condition}) {{\n"));
            render_cuda_stmt_list(out, body, indent + 1);
            out.push_str(&format!("{pad}}}\n"));
        }
        CudaStmt::Syncthreads => out.push_str(&format!("{pad}__syncthreads();\n")),
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
