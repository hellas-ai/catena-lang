use std::{collections::HashMap, fmt::Write};

use crate::compile::{
    cfg::{BlockInstruction, Cfg, CfgEdge, Transfer},
    graph_ops::Graph,
    program::{
        Context, Definition, DefinitionId, Program, Variable, VariableId as ProgramVariableId,
    },
};

use super::cfg::AnalysisCfg;

pub(super) fn render_analysis_cfg(graph: &Graph, analysis_cfg: AnalysisCfg) -> Vec<u8> {
    let AnalysisCfg {
        cfg,
        globals,
        block_svg_paths,
    } = analysis_cfg;
    let program = cfg_program(graph, cfg);
    let definition = program.entry_definition();
    let mut out = String::new();

    writeln!(&mut out, "definition {}", definition.name).expect("write to string cannot fail");
    writeln!(&mut out, "  parameters").expect("write to string cannot fail");
    for parameter in &definition.params {
        writeln!(&mut out, "    {}", render_variable(definition, *parameter))
            .expect("write to string cannot fail");
    }

    writeln!(&mut out, "  globals").expect("write to string cannot fail");
    for global in globals {
        writeln!(
            &mut out,
            "    {}",
            render_variable(definition, ProgramVariableId(global))
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
            render_wire_ids(definition, &node.params).join(", ")
        )
        .expect("write to string cannot fail");
        for instruction in &node.block {
            render_instruction(&mut out, definition, instruction);
        }
        render_transfer(&mut out, definition, &node.transfer);
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

fn render_instruction(out: &mut String, definition: &Definition, instruction: &BlockInstruction) {
    let results = render_wire_ids(definition, &instruction.results);
    let args = render_wire_ids(definition, &instruction.args);
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

fn render_transfer(out: &mut String, definition: &Definition, transfer: &Transfer) {
    match transfer {
        Transfer::Goto(edge) => {
            writeln!(out, "      goto {}", render_edge(definition, edge))
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
                render_wire_id(definition, *condition),
                render_edge(definition, then_edge),
                render_edge(definition, else_edge)
            )
            .expect("write to string cannot fail");
        }
        Transfer::Return(values) => {
            writeln!(
                out,
                "      return {}",
                render_wire_ids(definition, values).join(", ")
            )
            .expect("write to string cannot fail");
        }
    }
}

fn render_edge(definition: &Definition, edge: &CfgEdge) -> String {
    format!(
        "{}({})",
        definition.body.label(edge.target),
        render_wire_ids(definition, &edge.args).join(", ")
    )
}

fn render_variable(definition: &Definition, id: ProgramVariableId) -> String {
    definition
        .context
        .variable(id)
        .map(|variable| format!("{}: {}", variable.name, render_object(&variable.ty)))
        .unwrap_or_else(|| format!("w{}: <global>", id.0))
}

fn render_wire_ids(
    definition: &Definition,
    ids: &[crate::compile::cfg::VariableId],
) -> Vec<String> {
    ids.iter()
        .map(|id| render_wire_id(definition, *id))
        .collect()
}

fn render_wire_id(definition: &Definition, id: crate::compile::cfg::VariableId) -> String {
    definition
        .context
        .variable(ProgramVariableId(id))
        .map(|variable| variable.name.clone())
        .unwrap_or_else(|| crate::compile::cfg::variable_name(id))
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
