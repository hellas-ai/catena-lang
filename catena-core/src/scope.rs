/// Infer scope from open hypergraph structure
/// TODO: It may be possible to replace this with something having more theoretical support, e.g.,
/// the Danos–Regnier switching criterion for multiplicative linear logic.
// Algorithm sketch:
//  A two-pass algorithm,
//  Backwards pass: computes "longest common prefix" of reachable parent scopes
//  Forwards pass: computes shallowest possible scope in which each op can live
use metacat::ssa::{SSA, SSAError, ssa};
use open_hypergraphs::lax::{EdgeId, NodeId, OpenHypergraph};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::lang::{Arr, Obj};

/// Error computing scopes
#[derive(Debug, thiserror::Error)]
pub enum ScopeError {
    #[error("SSA Error: {0}")]
    SSAError(#[from] SSAError),

    #[error("Multiple scopes")]
    MultipleScopes,

    #[error("Incomplete Backwards pass")]
    IncompleteBackwardPass,
}

/// A uniquely identified scope.
/// Note that `EdgeId` is not sufficient, since one operation can have multiple associated
/// "blocks" (e.g., 'if b then t else f' has three).
#[derive(Hash, Clone, PartialEq, Eq, Debug)]
pub struct ScopeId {
    pub edge_id: EdgeId,
    pub scope_id: usize,
}

/// Compute the "inverse parent scope" relation: a mapping from each ScopeId to a *topologically
/// ordered* list of children.
pub fn scopes<O: Clone + PartialEq, A: Clone + Debug>(
    f: OpenHypergraph<O, A>,
    scope_ids: impl Fn(&A, &[&O], &[&O]) -> Vec<Option<usize>>,
) -> Result<HashMap<Option<ScopeId>, Vec<EdgeId>>, ScopeError> {
    // TODO: invert the parent scope relation, keeping SSA order.
    let n = f.hypergraph.nodes.len();
    let sources = f.sources.clone();
    let targets = f.targets.clone();

    let ssa = ssa(f.to_strict())?;

    // Propagate the 'stack' of parent scopes enclosing each node.
    // This computes the deepest scope a node must be available in.
    // TODO: should have no None values here(?)
    let bwd_stacks: Vec<_> = backward(scope_ids, n, &sources, &targets, &ssa)
        .into_iter()
        .collect::<Option<_>>()
        .ok_or(ScopeError::IncompleteBackwardPass)?;

    // Forward pass data initialize sources
    let mut fwd_stacks: Vec<Option<Stack>> = vec![None; n];
    for i in sources {
        fwd_stacks[i.0] = Some(Stack::new());
    }

    let mut result = HashMap::new();

    // Compute the shallowest possible scope into which we can place each op,
    // and order topologically
    for op in ssa.iter() {
        // Read each stack assigned to a source node, defaulting to data from the backward pass if
        // not present.
        let source_values = op
            .sources
            .iter()
            .map(|(i, _)| {
                fwd_stacks[i.0]
                    .clone()
                    .unwrap_or_else(|| bwd_stacks[i.0].clone())
            })
            .collect::<Vec<_>>();

        // Pop any trailing scopes belonging to this op from each source stack.
        // The op lives at the parent level; only its children are inside its scopes.
        let source_values: Vec<_> = source_values
            .into_iter()
            .map(|mut stack| {
                while let Some(last) = stack.last() {
                    if last.edge_id == op.edge_id {
                        stack.0.pop();
                    } else {
                        break;
                    }
                }
                stack
            })
            .collect();

        // The scope to which this operation belongs is the *shallowest*
        // read longest + default to empty stack
        // NOTE: all must be prefixes of the longest
        //  - [1, 2] ∧ [1, 3]
        let op_stack = Stack::longest(&source_values)?;

        // Update ops in op_scope to include this one
        result
            .entry(op_stack.last())
            .or_insert_with(Vec::new)
            .push(op.edge_id);

        // Write targets as op_scope
        for (i, _) in op.targets.iter() {
            fwd_stacks[i.0] = Some(op_stack.clone());
        }
    }

    Ok(result)
}

////////////////////////////////////////////////////////////////////////////////
// Private

#[derive(Clone, Debug, PartialEq, Default)]
struct Stack(Vec<ScopeId>);

/// Each operation computes the *longest common prefix* of each stack in its input nodes
/// TODO: add operation to determine if this op pushes to scope or not
fn backward<O, A>(
    scope_ids: impl Fn(&A, &[&O], &[&O]) -> Vec<Option<usize>>,
    n: usize,
    sources: &[NodeId],
    targets: &[NodeId],
    ssa: &[SSA<O, A>],
) -> Vec<Option<Stack>> {
    // Initialize targets to empty stacks, others to None
    // NOTE: None stands for the adjoined top element of the prefix order.
    let mut stacks: Vec<Option<Stack>> = vec![None; n];
    for t in targets {
        stacks[t.0] = Some(Stack::new());
    }

    // NOTE: sources must *also* be global, so no source can also be bound
    for s in sources {
        stacks[s.0] = Some(Stack::new());
    }

    // Propagate backwards: each op writes to its *sources* the longest common prefix of each
    // stack. None values are *ignored* since they are the "top" element e, where lcp(x, e) = x.
    for op in ssa.iter().rev() {
        // read stacks from target nodes
        let target_values = op
            .targets
            .iter()
            .filter_map(|(i, _)| stacks[i.0].clone())
            .collect();

        // Compute LCP
        let stack = Stack::longest_common_prefix(target_values);

        // Write result to *source* nodes
        let source_objects: Vec<_> = op.sources.iter().map(|(_, o)| o).collect();
        let target_objects: Vec<_> = op.targets.iter().map(|(_, o)| o).collect();
        let scopes = scope_ids(&op.op, &source_objects, &target_objects);

        // If an input is scoped, push a scope to its stack
        for ((i, _), scope_id) in op.sources.iter().zip(scopes) {
            let mut new_stack = stack.clone();
            if let Some(s) = scope_id {
                new_stack.push(ScopeId {
                    edge_id: op.edge_id,
                    scope_id: s,
                });
            }
            // If node already has a stack, take LCP to handle non-monogamous case
            stacks[i.0] = Some(match &stacks[i.0] {
                Some(existing) => Stack::longest_common_prefix(vec![existing.clone(), new_stack]),
                None => new_stack,
            });
        }
    }

    stacks
}

impl Stack {
    fn new() -> Self {
        Stack(vec![])
    }

    fn push(&mut self, e: ScopeId) {
        self.0.push(e)
    }

    fn last(&self) -> Option<ScopeId> {
        self.0.last().cloned()
    }

    // Return the longest stack and assert that all stacks are prefixes of the longest
    fn longest(xs: &[Self]) -> Result<Self, ScopeError> {
        if xs.is_empty() {
            return Ok(Stack::new());
        }

        // Find the longest stack
        let longest = xs.iter().max_by_key(|s| s.0.len()).unwrap();

        // Verify all others are prefixes of the longest
        for stack in xs {
            if !longest.0.starts_with(&stack.0) {
                return Err(ScopeError::MultipleScopes);
            }
        }

        Ok(longest.clone())
    }

    /// Compute the longest common prefix of a list of stacks
    fn longest_common_prefix(stacks: Vec<Self>) -> Self {
        if stacks.is_empty() {
            return Stack::new();
        }

        let mut result = stacks[0].0.clone();
        for stack in &stacks[1..] {
            // Truncate result to common prefix with this stack
            let common_len = result
                .iter()
                .zip(stack.0.iter())
                .take_while(|(a, b)| a == b)
                .count();
            result.truncate(common_len);
        }

        Stack(result)
    }
}

pub fn scope_ids(a: &Arr, sources: &[&Obj], _targets: &[&Obj]) -> Vec<Option<usize>> {
    match a.to_string().as_str() {
        "reduce" => vec![None, None, Some(0), Some(0), Some(0), Some(1), Some(1)],
        _ => vec![None; sources.len()],
    }
}
