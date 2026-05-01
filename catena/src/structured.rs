use crate::lang::{Arr, Obj};
use open_hypergraphs::lax::OpenHypergraph;

mod cuda;
pub mod ir;
mod ramsey;
mod tiled_matmul;

pub use ir::{EntryPoint, Param, Primitive, Program, Stmt};
pub use ramsey::StructuredError;

/// Operational pipeline for shallow hypergraph structuring:
///
/// 1. Interpret the shallow hypergraph as a control-flow graph. Hyperedges are
///    CFG nodes; a target wire followed by a consuming hyperedge is a CFG edge.
/// 2. Run Ramsey's "Beyond Relooper" recipe: reverse postorder, dominators,
///    immediate-dominator tree, merge-node detection, loop-header detection,
///    then recursive `do_tree`/`node_within`/`do_branch` structuring.
/// 3. Interpret opaque, declared-only arrows operationally. The generic Ramsey
///    layer knows only about structured control. The current tiled-matmul
///    mapping decides that selected prelude arrows are counted loops, selected
///    barrier arrows are synchronization, and primitive GPU arrows become C
///    snippets.
pub fn structured_from_shallow(
    f: &OpenHypergraph<Obj, Arr>,
    definition: &str,
) -> Result<Program, StructuredError> {
    let cfg = ramsey::Cfg::from_hypergraph(f)?;
    let semantics = tiled_matmul::TiledMatmulSemantics;
    let body = ramsey::structure(cfg, semantics)?;
    Ok(tiled_matmul::program(definition, body))
}
