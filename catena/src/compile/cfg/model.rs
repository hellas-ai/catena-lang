use crate::compile::{CompileGraph, CompileTheory, cfg::build::CfgBuilder};

pub type CfgNodeId = usize;
pub type OperationId = usize;
pub type OperationName = String;
pub type VariableId = usize;
pub type VariableName = String;

#[derive(Debug, thiserror::Error)]
pub enum CfgError {
    #[error("cfg only accepts data regions; got {0}")]
    UnsupportedTheory(CompileTheory),
    #[error("monoidal-structure wire `{wire}` produced by `{operation}` cannot resolve to an atom")]
    UnresolvedMonoidalStructureAtom {
        wire: VariableId,
        operation: OperationName,
    },
    #[error("cycle while resolving monoidal-structure wire `{0}`")]
    MonoidalStructureCycle(VariableId),
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
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
pub struct CfgNode {
    pub id: CfgNodeId,
    pub params: Vec<VariableId>,
    pub block: Vec<BlockInstruction>,
    pub transfer: Transfer,
}

#[derive(Debug, Clone)]
pub struct BlockInstruction {
    pub operation_id: OperationId,
    pub operation: OperationName,
    pub args: Vec<VariableId>,
    pub results: Vec<VariableId>,
}

#[derive(Debug, Clone)]
pub enum Transfer {
    Goto(CfgEdge),
    If {
        condition: VariableId,
        then_edge: CfgEdge,
        else_edge: CfgEdge,
    },
    Return(Vec<VariableId>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CfgEdge {
    pub target: CfgNodeId,
    pub args: Vec<VariableId>,
}

#[derive(Debug, Clone)]
pub(super) struct CfgNodeDraft {
    pub(super) id: CfgNodeId,
    pub(super) params: Vec<VariableId>,
    pub(super) block: Vec<BlockInstruction>,
}

#[derive(Debug, Clone)]
pub struct CfgWiring {
    pub node_boundaries: Vec<CfgNodeBoundaries>,
}

#[derive(Debug, Clone)]
pub struct CfgNodeBoundaries {
    pub node: CfgNodeId,
    pub entries: Vec<BoundaryPoint>,
    pub exits: Vec<BoundaryPoint>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BoundaryPoint {
    pub wire: VariableId,
    pub name: Option<VariableName>,
    pub kind: BoundaryKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoundaryKind {
    RegionEntry,
    RegionExit,
    FromControl(OperationId),
    ToControl(OperationId),
}

impl Cfg {
    pub fn from_compile_graph(compile_graph: &CompileGraph) -> Result<Self, CfgError> {
        Self::from_compile_graph_with_options(compile_graph, CfgOptions::default())
    }

    pub fn from_compile_graph_with_options(
        compile_graph: &CompileGraph,
        options: CfgOptions,
    ) -> Result<Self, CfgError> {
        CfgBuilder::new(compile_graph).with_options(options).build()
    }

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

pub fn variable_name(id: VariableId) -> String {
    if id > usize::MAX / 2 {
        format!("s{}", usize::MAX - id)
    } else {
        format!("w{id}")
    }
}
