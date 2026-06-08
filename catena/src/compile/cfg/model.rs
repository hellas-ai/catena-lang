use std::collections::HashMap;

use crate::compile::{CompileGraph, CompileTheory, cfg::layering::Layer};

pub(crate) type CfgNodeId = usize;
pub(crate) type OperationId = usize;
pub(crate) type VariableId = usize;

#[derive(Debug, thiserror::Error)]
pub enum CfgError {
    #[error("cfg only accepts data regions; got {0}")]
    UnsupportedTheory(CompileTheory),
}

#[derive(Debug, Clone)]
pub struct Cfg {
    pub(crate) entry: CfgNodeId,
    pub(crate) nodes: Vec<CfgNode>,
    pub(crate) predecessors: Vec<Vec<CfgNodeId>>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CfgOptions {
    pub keep_monoidal_operations: bool,
    pub keep_control_flow_operations: bool,
}

#[derive(Debug, Clone)]
pub struct CfgBuild {
    pub artifacts: CfgArtifacts,
    pub(crate) cfg: Cfg,
    pub(crate) globals: Vec<VariableId>,
    pub(crate) wire_names: HashMap<VariableId, String>,
    pub(crate) block_svg_paths: HashMap<CfgNodeId, String>,
}

impl CfgBuild {
    pub fn cfg(&self) -> &Cfg {
        &self.cfg
    }
}

#[derive(Debug, Clone)]
pub struct CfgArtifacts {
    pub(super) graph: CompileGraph,
    pub(super) layer: Layer,
    pub(super) cfg: Cfg,
    pub(super) globals: Vec<VariableId>,
    pub(super) wire_names: HashMap<VariableId, String>,
    pub(super) block_svg_paths: HashMap<CfgNodeId, String>,
}

#[derive(Debug, Clone)]
pub(crate) struct CfgNode {
    pub(crate) id: CfgNodeId,
    pub(crate) params: Vec<VariableId>,
    pub(crate) block: Vec<BlockInstruction>,
    pub(crate) transfer: Transfer,
}

#[derive(Debug, Clone)]
pub(crate) struct BlockInstruction {
    pub(crate) operation_id: OperationId,
    pub(crate) operation: String,
    pub(crate) args: Vec<VariableId>,
    pub(crate) results: Vec<VariableId>,
}

#[derive(Debug, Clone)]
pub(crate) enum Transfer {
    Goto(CfgEdge),
    If {
        condition: VariableId,
        then_edge: CfgEdge,
        else_edge: CfgEdge,
    },
    Return(Vec<VariableId>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CfgEdge {
    pub(crate) target: CfgNodeId,
    pub(crate) args: Vec<VariableId>,
}

impl Cfg {
    pub(crate) fn label(&self, node: CfgNodeId) -> String {
        format!("n{node}")
    }
}
impl CfgNode {
    pub(crate) fn successors(&self) -> Vec<CfgNodeId> {
        match &self.transfer {
            Transfer::Goto(edge) => vec![edge.target],
            Transfer::If {
                then_edge,
                else_edge,
                ..
            } => vec![then_edge.target, else_edge.target],
            Transfer::Return(_) => Vec::new(),
        }
    }
}

// Variable naming

pub(crate) fn variable_name(id: VariableId) -> String {
    if id > usize::MAX / 2 {
        format!("s{}", usize::MAX - id)
    } else {
        format!("w{id}")
    }
}
