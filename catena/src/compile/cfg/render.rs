use metacat::tree::Tree;

use crate::{
    compile::{
        cfg::{BlockInstruction, CfgEdge, Transfer},
        program::{Definition, Program, VariableId},
    },
    lang::Obj,
};

pub fn render_program_cfg(program: &Program) -> String {
    let mut out = String::new();
    let mut definitions = program.definitions.values().collect::<Vec<_>>();
    definitions.sort_by_key(|definition| definition.id.0);
    for (index, definition) in definitions.iter().enumerate() {
        if index > 0 {
            out.push('\n');
        }
        render_definition_cfg(&mut out, definition);
    }
    out
}

fn render_definition_cfg(out: &mut String, definition: &Definition) {
    out.push_str(&format!("definition {}\n", definition.name));
    out.push_str("  parameters\n");
    for parameter in &definition.params {
        out.push_str(&format!(
            "    {}\n",
            render_variable(definition, *parameter)
        ));
    }
    out.push_str(&format!(
        "  entry {}\n",
        definition.body.label(definition.body.entry)
    ));
    out.push_str("  blocks\n");

    for node in &definition.body.nodes {
        out.push_str(&format!(
            "    {}({})\n",
            definition.body.label(node.id),
            render_wire_ids(definition, &node.params).join(", ")
        ));
        for instruction in &node.block {
            render_instruction(out, definition, instruction);
        }
        render_transfer(out, definition, &node.transfer);
    }
}

fn render_instruction(out: &mut String, definition: &Definition, instruction: &BlockInstruction) {
    let results = render_wire_ids(definition, &instruction.results);
    let args = render_wire_ids(definition, &instruction.args);
    if results.is_empty() {
        out.push_str(&format!(
            "      {}#{}({})\n",
            instruction.operation,
            instruction.operation_id,
            args.join(", ")
        ));
    } else {
        out.push_str(&format!(
            "      {} = {}#{}({})\n",
            results.join(", "),
            instruction.operation,
            instruction.operation_id,
            args.join(", ")
        ));
    }
}

fn render_transfer(out: &mut String, definition: &Definition, transfer: &Transfer) {
    match transfer {
        Transfer::Goto(edge) => {
            out.push_str(&format!("      goto {}\n", render_edge(definition, edge)));
        }
        Transfer::If {
            condition,
            then_edge,
            else_edge,
        } => {
            out.push_str(&format!(
                "      if {} then {} else {}\n",
                render_wire_id(definition, *condition),
                render_edge(definition, then_edge),
                render_edge(definition, else_edge)
            ));
        }
        Transfer::Return(values) => {
            out.push_str(&format!(
                "      return {}\n",
                render_wire_ids(definition, values).join(", ")
            ));
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

fn render_variable(definition: &Definition, id: VariableId) -> String {
    definition
        .context
        .variable(id)
        .map(|variable| format!("{}: {}", variable.name, render_object(&variable.ty)))
        .unwrap_or_else(|| format!("w{}: <unknown>", id.0))
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
        .variable(VariableId(id))
        .map(|variable| variable.name.clone())
        .unwrap_or_else(|| crate::compile::cfg::variable_name(id))
}

fn render_object(object: &Obj) -> String {
    match object {
        Tree::Empty => "empty".to_string(),
        Tree::Leaf(index, _) => format!("x{index}"),
        Tree::Node(op, target_index, children) => {
            let inner = render_object_node(op, children);
            if *target_index == 0 {
                inner
            } else {
                format!("proj{target_index}({inner})")
            }
        }
    }
}

fn render_object_node(op: &hexpr::Operation, children: &[Obj]) -> String {
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
