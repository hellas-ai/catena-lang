mod build;
mod control;
mod data;
mod model;
#[allow(dead_code)]
mod monoidal;
mod operation;
mod render;
mod wiring;

pub use model::{
    BlockInstruction, BoundaryKind, BoundaryPoint, Cfg, CfgEdge, CfgError, CfgNode,
    CfgNodeBoundaries, CfgNodeId, CfgOptions, CfgWiring, OperationId, OperationName, Transfer,
    VariableId, VariableName, variable_name,
};
pub use render::render_program_cfg;
