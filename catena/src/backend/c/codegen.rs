use crate::lang::*;
use crate::scope::{ScopeId, scope_ids, scopes};

use metacat::ssa::{SSA, ssa};
use metacat::theory::OperationKey;
use metacat::tree::Tree;

use open_hypergraphs::lax::{EdgeId, OpenHypergraph};
use std::collections::HashMap;

#[derive(Debug)]
pub enum CodegenError {
    InvalidHexpr(String), // An Internal programmer error
    NoCType(Obj),
}

/// Render a term as C code
pub fn codegen(f: OpenHypergraph<Obj, Arr>, fn_name: &str) -> String {
    // scope_map: Option<ScopeId> -> Vec<EdgeId> (topologically ordered)
    let scope_map = scopes(f.clone(), scope_ids).unwrap();

    // Build EdgeId -> SSA lookup
    // NOTE: should probably change `scopes` so it returns SSA info, not just edge ID
    let ssa_list = ssa(f.clone().to_strict()).unwrap();
    let ssa_map: HashMap<EdgeId, SSA<Obj, Arr>> =
        ssa_list.into_iter().map(|op| (op.edge_id, op)).collect();

    // Render top-level operations
    let top_level = scope_map.get(&None).unwrap();
    let mut statements = vec![];
    for edge_id in top_level {
        statements.push(render_c(&ssa_map[edge_id], &scope_map, &ssa_map));
    }

    let body = statements.join("\n");
    format_function(&f, fn_name, &body)
}

// Using the sources/targets of `f : m -> n`, generate a function prelude like the below
//
// ```c
// void fn_name(type_s0 v0, ..., type_sm vm, ..., type_t0* r0, ... type_tn rn) {
//     {body}
//     {postlude}
// }
// ```
//
// Note that multiple returns are modeled as *pointers* arguments to the n return values.
pub fn format_function(f: &OpenHypergraph<Obj, Arr>, fn_name: &str, body: &str) -> String {
    let mut args: Vec<(String, String)> = f
        .sources
        .iter()
        .map(|node_id| {
            let obj = &f.hypergraph.nodes[node_id.0];
            (to_c_type(obj).unwrap(), format!("v{}", node_id.0))
        })
        .collect();

    // (TypeName, Var) pairs.
    // Vars named r{i} for each target i.
    args.extend(f.targets.iter().map(|node_id| {
        let obj = &f.hypergraph.nodes[node_id.0];
        (
            format!("{}*", to_c_type(obj).unwrap()),
            format!("r{}", node_id.0),
        )
    }));

    // All comma separated args (sources and targets)
    let args = args
        .into_iter()
        .map(|(t, v)| format!("{} {}", t, v))
        .collect::<Vec<_>>()
        .join(", ");

    // *r{i} = v{i} for each i in targets
    let postlude_statements: Vec<String> = f
        .targets
        .iter()
        .map(|node_id| format!("*r{} = v{};", node_id.0, node_id.0))
        .collect();
    let postlude = postlude_statements.join("\n");

    format!(
        r#"
        void {fn_name}({args}) {{
            {body}
            {postlude}
        }}
        "#
    )
    .trim_start_matches(' ')
    .to_string()
}

/// TODO: this is recursive, but we should use a stack instead
pub fn render_c(
    op: &SSA<Obj, Arr>,
    scope_map: &HashMap<Option<ScopeId>, Vec<EdgeId>>,
    ssa_map: &HashMap<EdgeId, SSA<Obj, Arr>>,
) -> String {
    match op.op.to_string().as_str() {
        "f32.zero" => {
            let t0 = &op.targets[0].0.0;
            format!("float v{t0} = 0.0; // f32.zero")
        }
        "f32.one" => {
            let t0 = &op.targets[0].0.0;
            format!("float v{t0} = 1.0; // f32.one")
        }
        "f32.add" => {
            let s0 = &op.sources[0].0.0;
            let s1 = &op.sources[1].0.0;
            let t0 = &op.targets[0].0.0;
            format!("float v{t0} = v{s0} + v{s1}; // f32.add")
        }
        "f32.mul" => {
            let s0 = &op.sources[0].0.0;
            let s1 = &op.sources[1].0.0;
            let t0 = &op.targets[0].0.0;
            format!("float v{t0} = v{s0} + v{s1}; // f32.add")
        }
        "f32.from-index" => {
            let s0 = &op.sources[0].0.0;
            let t0 = &op.targets[0].0.0;
            format!("float v{t0} = (float) v{s0}; // f32.from-index")
        }
        "reduce" => {
            // Sources: [Extent, Zero, Acc, Val, Combined, Index*, Body]
            let extent = &op.sources[0].0.0;
            let zero = &op.sources[1].0.0;
            let acc = &op.sources[2].0.0;
            let val = &op.sources[3].0.0;
            let combined = &op.sources[4].0.0;
            let index = &op.sources[5].0.0;
            let body = &op.sources[6].0.0;

            let result = &op.targets[0].0.0;

            // Body scope (scope 1): computes body value from index
            let body_scope_id = ScopeId {
                edge_id: op.edge_id,
                scope_id: 1,
            };
            let body_statements = scope_map
                .get(&Some(body_scope_id))
                .into_iter()
                .flatten()
                .map(|edge_id| render_c(&ssa_map[edge_id], scope_map, ssa_map))
                .collect::<Vec<_>>()
                .join("\n");

            // Combine scope (scope 0): combines acc with val into combined
            let combine_scope_id = ScopeId {
                edge_id: op.edge_id,
                scope_id: 0,
            };
            let combine_statements = scope_map
                .get(&Some(combine_scope_id))
                .into_iter()
                .flatten()
                .map(|edge_id| render_c(&ssa_map[edge_id], scope_map, ssa_map))
                .collect::<Vec<_>>()
                .join("\n");

            format!(
                r#"

                //// START REDUCE ////
                // v{result} = reduce
                float v{result} = v{zero}; // result = zero
                for(uint64_t v{index} = 0; v{index} < v{extent}; v{index}++) {{
                {body_statements}
                    float v{acc} = v{result}; // acc = result
                    float v{val} = v{body}; // val = body
                    {combine_statements}
                    v{result} = v{combined}; // result = combined
                }}
                //// END REDUCE ////
                "#
            )
            .trim_start_matches('\n')
            .to_string()
        }
        s => panic!("unknown builtin: {s}"),
    }
}

// Basic pattern matching to map value types to C types
// NOTE: this is quite limited, and will only work for types of the form value(<name>(...)).
// But it's sufficient for now.
fn to_c_type(obj: &Obj) -> Result<String, CodegenError> {
    match_value_types(
        obj,
        &[
            ("f32", "float"),
            ("index", "uint64_t"),
            ("extent", "uint64_t"),
        ],
    )
    .map(|x| x.to_string())
    .ok_or_else(|| CodegenError::NoCType(obj.clone()))
}

fn match_value_types<T: Clone>(tree: &Tree<(), OperationKey>, cases: &[(&str, T)]) -> Option<T> {
    for (name, value) in cases {
        if match_value_type(name, tree) {
            return Some(value.clone());
        }
    }
    None
}

// Match any tree of the form π0(value(π0(<name>(...))))
fn match_value_type(name: &str, tree: &Tree<(), OperationKey>) -> bool {
    // Top level must be a 'value' node
    match tree {
        Tree::Node(val, 0, children) if val.to_string() == "value" => {
            // match a single child
            let [Tree::Node(key, 0, _)] = children.as_slice() else {
                return false;
            };

            key.to_string() == name
        }
        _ => false,
    }
}
