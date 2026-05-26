use std::collections::{HashMap, HashSet};

use crate::{
    compile::cuda::resources::SharedIndexing,
    structured::ir::{Primitive, Stmt, StructuredProgram},
};

// View values are normally flat CUDA indices. Static shared memory is the
// exception: a declaration like `__shared__ float tile[R][C]` must be indexed
// with structured coordinates. This analysis finds the views used with static
// shared resources so domain lowering can emit `view_x`, `view_y`, and `view_z`
// side variables only where they are needed.
#[derive(Debug, Clone)]
pub(super) struct ViewAnalysis {
    shared_indexing: HashMap<String, SharedIndexing>,
    static_view_ranks: HashMap<String, usize>,
}

impl ViewAnalysis {
    pub(super) fn new(
        program: &StructuredProgram,
        names: &HashMap<String, String>,
        mut shared_indexing: HashMap<String, SharedIndexing>,
    ) -> Self {
        collect_shared_aliases(&program.body, names, &mut shared_indexing);
        let mut static_view_ranks = HashMap::new();
        collect_static_shared_views(
            &program.body,
            names,
            &shared_indexing,
            &mut static_view_ranks,
        );
        Self {
            shared_indexing,
            static_view_ranks,
        }
    }

    pub(super) fn shared_access(&self, shared: &str, view: &str) -> String {
        match self.shared_indexing.get(shared) {
            Some(SharedIndexing::Static { rank: 1 }) => {
                format!("{shared}[{}]", view_component(view, "x"))
            }
            Some(SharedIndexing::Static { rank: 2 }) => {
                format!(
                    "{shared}[{}][{}]",
                    view_component(view, "x"),
                    view_component(view, "y")
                )
            }
            Some(SharedIndexing::Static { rank: 3 }) => {
                format!(
                    "{shared}[{}][{}][{}]",
                    view_component(view, "x"),
                    view_component(view, "y"),
                    view_component(view, "z")
                )
            }
            _ => format!("{shared}[{view}]"),
        }
    }

    pub(super) fn static_view_rank(&self, view: &str) -> Option<usize> {
        self.static_view_ranks.get(view).copied()
    }
}

pub(super) fn extents_required_by_device_code(program: &StructuredProgram) -> HashSet<String> {
    let mut names = HashSet::new();
    collect_extents_required_by_device_code(&program.body, &mut names);
    names
}

fn view_component(view: &str, component: &str) -> String {
    format!("{view}_{component}")
}

fn collect_shared_aliases(
    stmts: &[Stmt],
    names: &HashMap<String, String>,
    shared_indexing: &mut HashMap<String, SharedIndexing>,
) {
    for stmt in stmts {
        match stmt {
            Stmt::Block { body, .. } | Stmt::Loop { body, .. } | Stmt::For { body, .. } => {
                collect_shared_aliases(body, names, shared_indexing);
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                collect_shared_aliases(then_body, names, shared_indexing);
                collect_shared_aliases(else_body, names, shared_indexing);
            }
            Stmt::Switch { cases, .. } => {
                for case in cases {
                    collect_shared_aliases(case, names, shared_indexing);
                }
            }
            Stmt::Primitive(primitive) if primitive.name == "gpu.shared.store" => {
                let Some(shared) = primitive.inputs.first() else {
                    continue;
                };
                let Some(output) = primitive.outputs.first() else {
                    continue;
                };
                let shared = names.get(shared).unwrap_or(shared);
                let output = names.get(output).unwrap_or(output);
                if let Some(indexing) = shared_indexing.get(shared).cloned() {
                    shared_indexing.insert(output.clone(), indexing);
                }
            }
            Stmt::Primitive(_)
            | Stmt::Break(_)
            | Stmt::Continue(_)
            | Stmt::Return
            | Stmt::Barrier
            | Stmt::Assign { .. }
            | Stmt::Comment(_) => {}
        }
    }
}

fn collect_static_shared_views(
    stmts: &[Stmt],
    names: &HashMap<String, String>,
    shared_indexing: &HashMap<String, SharedIndexing>,
    static_view_ranks: &mut HashMap<String, usize>,
) {
    for stmt in stmts {
        match stmt {
            Stmt::Block { body, .. } | Stmt::Loop { body, .. } | Stmt::For { body, .. } => {
                collect_static_shared_views(body, names, shared_indexing, static_view_ranks);
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                collect_static_shared_views(then_body, names, shared_indexing, static_view_ranks);
                collect_static_shared_views(else_body, names, shared_indexing, static_view_ranks);
            }
            Stmt::Switch { cases, .. } => {
                for case in cases {
                    collect_static_shared_views(case, names, shared_indexing, static_view_ranks);
                }
            }
            Stmt::Primitive(primitive)
                if primitive.name == "gpu.shared.load" || primitive.name == "gpu.shared.store" =>
            {
                let Some(shared) = primitive.inputs.first() else {
                    continue;
                };
                let Some(view) = primitive.inputs.get(1) else {
                    continue;
                };
                let shared = names.get(shared).unwrap_or(shared);
                if let Some(SharedIndexing::Static { rank }) = shared_indexing.get(shared) {
                    static_view_ranks.insert(view.clone(), *rank);
                }
            }
            Stmt::Primitive(_)
            | Stmt::Break(_)
            | Stmt::Continue(_)
            | Stmt::Return
            | Stmt::Barrier
            | Stmt::Assign { .. }
            | Stmt::Comment(_) => {}
        }
    }
}

fn collect_extents_required_by_device_code(stmts: &[Stmt], names: &mut HashSet<String>) {
    for stmt in stmts {
        match stmt {
            Stmt::Block { body, .. } | Stmt::Loop { body, .. } | Stmt::For { body, .. } => {
                collect_extents_required_by_device_code(body, names);
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                collect_extents_required_by_device_code(then_body, names);
                collect_extents_required_by_device_code(else_body, names);
            }
            Stmt::Switch { cases, .. } => {
                for case in cases {
                    collect_extents_required_by_device_code(case, names);
                }
            }
            Stmt::Primitive(primitive) => {
                collect_primitive_extents_required_by_device_code(primitive, names);
            }
            Stmt::Break(_)
            | Stmt::Continue(_)
            | Stmt::Return
            | Stmt::Barrier
            | Stmt::Assign { .. }
            | Stmt::Comment(_) => {}
        }
    }
}

fn collect_primitive_extents_required_by_device_code(
    primitive: &Primitive,
    names: &mut HashSet<String>,
) {
    if primitive.name == "gpu.view.group-by-tile" {
        for extent in primitive.inputs.iter().skip(1).take(2) {
            names.insert(extent.clone());
        }
    }
    if primitive.name == "gpu.view.group"
        && let Some(cols) = primitive.inputs.get(1)
    {
        names.insert(cols.clone());
    }
    if primitive.name == "gpu.shape.row" || primitive.name == "gpu.shape.col" {
        if let Some(extent) = primitive.inputs.first() {
            names.insert(extent.clone());
        }
    }
    if primitive.name == "gpu.shape.row-mul" || primitive.name == "gpu.shape.col-mul" {
        for extent in primitive.inputs.iter().take(2) {
            names.insert(extent.clone());
        }
    }
    if primitive.name == "gpu.shape.2d" {
        for extent in primitive.inputs.iter().take(2) {
            names.insert(extent.clone());
        }
    }
}
