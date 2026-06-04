use std::collections::HashMap;

use crate::compile::{
    cfg::{Cfg, render_program_cfg_with_block_annotations},
    graph_ops::Graph,
    program::{
        Context, Definition, DefinitionId, Program, Variable, VariableId as ProgramVariableId,
    },
};

use super::cfg::AnalysisCfg;

pub(super) fn render_analysis_cfg(graph: &Graph, analysis_cfg: AnalysisCfg) -> Vec<u8> {
    let AnalysisCfg {
        cfg,
        block_svg_paths,
    } = analysis_cfg;
    render_program_cfg_with_block_annotations(&cfg_program(graph, cfg), |node| {
        block_svg_paths.get(&node).map(String::as_str)
    })
    .into_bytes()
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
