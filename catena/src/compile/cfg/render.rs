use std::{collections::HashMap, fmt::Write};

use crate::compile::{
    cfg::{BlockInstruction, Cfg, CfgBuild, CfgEdge, CfgNodeId, Transfer, VariableId},
    graph_ops::Graph,
    program::{
        Context, Definition, DefinitionId, Program, Variable, VariableId as ProgramVariableId,
    },
};

pub(super) fn render_cfg_build(cfg_build: &CfgBuild) -> Vec<u8> {
    render_cfg_parts(
        &cfg_build.artifacts.graph.graph,
        &cfg_build.cfg,
        &cfg_build.globals,
        &cfg_build.wire_names,
        &cfg_build.block_svg_paths,
    )
}

pub(super) fn render_cfg_parts(
    graph: &Graph,
    cfg: &Cfg,
    globals: &[VariableId],
    wire_names: &HashMap<VariableId, String>,
    block_svg_paths: &HashMap<CfgNodeId, String>,
) -> Vec<u8> {
    let program = cfg_program(graph, cfg.clone());
    let definition = program.entry_definition();
    let mut out = String::new();

    writeln!(&mut out, "definition {}", definition.name).expect("write to string cannot fail");
    writeln!(&mut out, "  parameters").expect("write to string cannot fail");
    for parameter in &definition.params {
        writeln!(
            &mut out,
            "    {}",
            render_variable(definition, wire_names, *parameter)
        )
        .expect("write to string cannot fail");
    }

    writeln!(&mut out, "  globals").expect("write to string cannot fail");
    for global in globals {
        writeln!(
            &mut out,
            "    {}",
            render_variable(definition, wire_names, ProgramVariableId(*global))
        )
        .expect("write to string cannot fail");
    }

    writeln!(
        &mut out,
        "  entry {}",
        definition.body.label(definition.body.entry)
    )
    .expect("write to string cannot fail");
    writeln!(&mut out, "  blocks").expect("write to string cannot fail");

    for node in &definition.body.nodes {
        write!(&mut out, "    {}", definition.body.label(node.id))
            .expect("write to string cannot fail");
        if let Some(annotation) = block_svg_paths.get(&node.id) {
            write!(&mut out, " [{annotation}]").expect("write to string cannot fail");
        }
        writeln!(
            &mut out,
            "({})",
            render_wire_ids(definition, wire_names, &node.params).join(", ")
        )
        .expect("write to string cannot fail");
        for instruction in &node.block {
            render_instruction(&mut out, definition, wire_names, instruction);
        }
        render_transfer(&mut out, definition, wire_names, &node.transfer);
    }

    out.into_bytes()
}

fn cfg_program(graph: &Graph, body: Cfg) -> Program {
    let entry = DefinitionId(0);
    Program {
        entry,
        definitions: HashMap::from([(
            entry,
            Definition {
                id: entry,
                name: "cfg".to_string(),
                params: graph
                    .s
                    .table
                    .iter()
                    .map(|wire| ProgramVariableId(*wire))
                    .collect(),
                returns: graph
                    .t
                    .table
                    .iter()
                    .map(|wire| ProgramVariableId(*wire))
                    .collect(),
                context: context_for_graph(graph),
                body,
            },
        )]),
    }
}

fn context_for_graph(graph: &Graph) -> Context {
    Context::new(
        graph
            .h
            .w
            .0
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, ty)| {
                (
                    ProgramVariableId(index),
                    Variable {
                        id: ProgramVariableId(index),
                        name: crate::compile::cfg::variable_name(index),
                        ty,
                    },
                )
            })
            .collect(),
    )
}

fn render_instruction(
    out: &mut String,
    definition: &Definition,
    wire_names: &HashMap<usize, String>,
    instruction: &BlockInstruction,
) {
    let results = render_wire_ids(definition, wire_names, &instruction.results);
    let args = render_wire_ids(definition, wire_names, &instruction.args);
    if results.is_empty() {
        writeln!(
            out,
            "      {}#{}({})",
            instruction.operation,
            instruction.operation_id,
            args.join(", ")
        )
        .expect("write to string cannot fail");
    } else {
        writeln!(
            out,
            "      {} = {}#{}({})",
            results.join(", "),
            instruction.operation,
            instruction.operation_id,
            args.join(", ")
        )
        .expect("write to string cannot fail");
    }
}

fn render_transfer(
    out: &mut String,
    definition: &Definition,
    wire_names: &HashMap<usize, String>,
    transfer: &Transfer,
) {
    match transfer {
        Transfer::Goto(edge) => {
            writeln!(
                out,
                "      goto {}",
                render_edge(definition, wire_names, edge)
            )
            .expect("write to string cannot fail");
        }
        Transfer::If {
            condition,
            then_edge,
            else_edge,
        } => {
            writeln!(
                out,
                "      if {} then {} else {}",
                render_wire_id(definition, wire_names, *condition),
                render_edge(definition, wire_names, then_edge),
                render_edge(definition, wire_names, else_edge)
            )
            .expect("write to string cannot fail");
        }
        Transfer::Return(values) => {
            writeln!(
                out,
                "      return {}",
                render_wire_ids(definition, wire_names, values).join(", ")
            )
            .expect("write to string cannot fail");
        }
    }
}

fn render_edge(
    definition: &Definition,
    wire_names: &HashMap<usize, String>,
    edge: &CfgEdge,
) -> String {
    format!(
        "{}({})",
        definition.body.label(edge.target),
        render_wire_ids(definition, wire_names, &edge.args).join(", ")
    )
}

fn render_variable(
    definition: &Definition,
    wire_names: &HashMap<usize, String>,
    id: ProgramVariableId,
) -> String {
    let rendered_id = render_wire_id(definition, wire_names, id.0);
    definition
        .context
        .variable(id)
        .map(|variable| format!("{rendered_id}: {}", render_object(&variable.ty)))
        .unwrap_or_else(|| format!("{rendered_id}: <global>"))
}

fn render_wire_ids(
    definition: &Definition,
    wire_names: &HashMap<usize, String>,
    ids: &[crate::compile::cfg::VariableId],
) -> Vec<String> {
    ids.iter()
        .map(|id| render_wire_id(definition, wire_names, *id))
        .collect()
}

fn render_wire_id(
    definition: &Definition,
    wire_names: &HashMap<usize, String>,
    id: crate::compile::cfg::VariableId,
) -> String {
    let wire = definition
        .context
        .variable(ProgramVariableId(id))
        .map(|variable| variable.name.clone())
        .unwrap_or_else(|| crate::compile::cfg::variable_name(id));
    match wire_names.get(&id) {
        Some(name) => format!("{wire} /* {} */", render_wire_name_annotation(name)),
        None => wire,
    }
}

fn render_wire_name_annotation(name: &str) -> String {
    name.replace("*/", "* /")
}

fn render_object(object: &crate::lang::Obj) -> String {
    match object {
        metacat::tree::Tree::Empty => "empty".to_string(),
        metacat::tree::Tree::Leaf(index, _) => format!("x{index}"),
        metacat::tree::Tree::Node(op, target_index, children) => {
            let inner = render_object_node(op, children);
            if *target_index == 0 {
                inner
            } else {
                format!("proj{target_index}({inner})")
            }
        }
    }
}

fn render_object_node(op: &hexpr::Operation, children: &[crate::lang::Obj]) -> String {
    match children {
        [] => op.to_string(),
        [child] => format!("{op}({})", render_object(child)),
        _ => {
            let args = children
                .iter()
                .map(render_object)
                .collect::<Vec<_>>()
                .join(", ");
            format!("{op}({args})")
        }
    }
}
