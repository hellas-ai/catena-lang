use std::fmt::Write;

use super::{Cfg, CfgEdge, CfgNode, Transfer, VariableId, variable_name};

pub fn render_cfg(cfg: &Cfg) -> String {
    let mut output = String::new();
    writeln!(output, "cfg entry n{}", cfg.entry).expect("write to string cannot fail");
    output.push('\n');
    for node in &cfg.nodes {
        render_node(&mut output, node);
    }
    output
}

fn render_node(output: &mut String, node: &CfgNode) {
    writeln!(output, "node n{}:", node.id).expect("write to string cannot fail");
    writeln!(output, "  params: {}", render_variables(&node.params))
        .expect("write to string cannot fail");
    output.push_str("  block:\n");
    if node.block.is_empty() {
        output.push_str("    <empty>\n");
    } else {
        for instruction in &node.block {
            writeln!(
                output,
                "    #{} {}({}) -> {}",
                instruction.operation_id,
                instruction.operation,
                render_variables(&instruction.args),
                render_variables(&instruction.results),
            )
            .expect("write to string cannot fail");
        }
    }
    writeln!(output, "  transfer: {}", render_transfer(&node.transfer))
        .expect("write to string cannot fail");
    output.push('\n');
}

fn render_transfer(transfer: &Transfer) -> String {
    match transfer {
        Transfer::Goto(edge) => format!("goto {}", render_edge(edge)),
        Transfer::If {
            condition,
            then_edge,
            else_edge,
        } => format!(
            "if {} then {} else {}",
            variable_name(*condition),
            render_edge(then_edge),
            render_edge(else_edge)
        ),
        Transfer::Return(results) => format!("return {}", render_variables(results)),
    }
}

fn render_edge(edge: &CfgEdge) -> String {
    format!("n{}({})", edge.target, render_variables(&edge.args))
}

fn render_variables(variables: &[VariableId]) -> String {
    variables
        .iter()
        .copied()
        .map(variable_name)
        .collect::<Vec<_>>()
        .join(", ")
}
