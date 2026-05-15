#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredProgram {
    pub name: String,
    pub entry: EntryPoint,
    pub body: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryPoint {
    pub name: String,
    pub params: Vec<Param>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    pub ty: String,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    Block {
        label: String,
        body: Vec<Stmt>,
    },
    Loop {
        label: String,
        body: Vec<Stmt>,
    },
    For {
        label: String,
        var: String,
        extent: String,
        body: Vec<Stmt>,
    },
    If {
        condition: String,
        then_body: Vec<Stmt>,
        else_body: Vec<Stmt>,
    },
    Switch {
        selector: String,
        cases: Vec<Vec<Stmt>>,
    },
    Break(String),
    Continue(String),
    Return,
    Barrier,
    Assign {
        lhs: String,
        rhs: String,
    },
    Primitive(Primitive),
    Comment(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Primitive {
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub code: String,
}

impl StructuredProgram {
    pub fn render_ir(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("program {}(", self.name));
        out.push_str(
            &self
                .entry
                .params
                .iter()
                .map(|p| format!("{} {}", p.ty, p.name))
                .collect::<Vec<_>>()
                .join(", "),
        );
        out.push_str(")\n");
        render_ir_stmts(&mut out, &self.body, 1);
        out
    }
}

fn render_ir_stmts(out: &mut String, stmts: &[Stmt], indent: usize) {
    let pad = "  ".repeat(indent);
    for stmt in stmts {
        match stmt {
            Stmt::Block { label, body } => {
                out.push_str(&format!("{pad}block {label}\n"));
                render_ir_stmts(out, body, indent + 1);
                out.push_str(&format!("{pad}end\n"));
            }
            Stmt::Loop { label, body } => {
                out.push_str(&format!("{pad}loop {label}\n"));
                render_ir_stmts(out, body, indent + 1);
                out.push_str(&format!("{pad}end\n"));
            }
            Stmt::For {
                var, extent, body, ..
            } => {
                out.push_str(&format!("{pad}for {var} in 0..{extent}\n"));
                render_ir_stmts(out, body, indent + 1);
                out.push_str(&format!("{pad}end\n"));
            }
            Stmt::If {
                condition,
                then_body,
                else_body,
            } => {
                out.push_str(&format!("{pad}if {condition}\n"));
                render_ir_stmts(out, then_body, indent + 1);
                out.push_str(&format!("{pad}else\n"));
                render_ir_stmts(out, else_body, indent + 1);
                out.push_str(&format!("{pad}end\n"));
            }
            Stmt::Switch { selector, cases } => {
                out.push_str(&format!("{pad}switch {selector}\n"));
                for (index, body) in cases.iter().enumerate() {
                    out.push_str(&format!("{pad}case {index}\n"));
                    render_ir_stmts(out, body, indent + 1);
                }
                out.push_str(&format!("{pad}end\n"));
            }
            Stmt::Break(label) => out.push_str(&format!("{pad}break {label}\n")),
            Stmt::Continue(label) => out.push_str(&format!("{pad}continue {label}\n")),
            Stmt::Return => out.push_str(&format!("{pad}return\n")),
            Stmt::Barrier => out.push_str(&format!("{pad}barrier\n")),
            Stmt::Assign { lhs, rhs } => out.push_str(&format!("{pad}{lhs} = {rhs}\n")),
            Stmt::Primitive(primitive) => {
                if primitive.outputs.is_empty() {
                    if primitive.inputs.is_empty() {
                        out.push_str(&format!("{pad}{}\n", primitive.name));
                    } else {
                        out.push_str(&format!(
                            "{pad}{}({})\n",
                            primitive.name,
                            primitive.inputs.join(", ")
                        ));
                    }
                } else {
                    out.push_str(&format!(
                        "{pad}{} = {}({})\n",
                        primitive.outputs.join(", "),
                        primitive.name,
                        primitive.inputs.join(", ")
                    ));
                }
            }
            Stmt::Comment(comment) => out.push_str(&format!("{pad}// {comment}\n")),
        }
    }
}
