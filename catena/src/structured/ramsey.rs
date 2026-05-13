use super::ir::Stmt;
use crate::lang::{Arr, Obj};
use open_hypergraphs::lax::{NodeId, OpenHypergraph};
use std::collections::{BTreeSet, HashMap, HashSet};

pub type CfgNodeId = usize;
pub type Expr = String;
pub type OperationName = String;
pub type SsaValue = String;

pub trait ArrowSemantics {
    fn statements(&self, arrow: &ArrowInstance) -> Vec<Stmt>;

    fn branch_condition_rhs(&self, arrow: &ArrowInstance, output: usize) -> Expr {
        format!("/* {} output {output} */ 1", sanitize_ident(&arrow.op))
    }

    fn selector(&self, arrow: &ArrowInstance) -> SsaValue {
        format!("/* {} */ 0", sanitize_ident(&arrow.op))
    }

    fn counted_loop(&self, _op: &str) -> Option<(SsaValue, Expr)> {
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrowInstance {
    pub id: CfgNodeId,
    pub op: OperationName,
    pub inputs: Vec<SsaValue>,
    pub outputs: Vec<SsaValue>,
    pub branch_arity: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum StructuredError {
    #[error("shallow graph has no operation reachable from the source interface")]
    MissingEntry,
    #[error("control-flow graph has an irreducible back edge from {from} to {to}")]
    IrreducibleBackEdge { from: String, to: String },
    #[error("branch target {0} is not in the structured context")]
    MissingContext(String),
    #[error("control node {node} has {successors} entry successors; only one entry is supported")]
    UnsupportedEntry { node: String, successors: usize },
}

pub fn structure(cfg: Cfg) -> Result<Vec<Stmt>, StructuredError> {
    let analyses = Analyses::new(&cfg)?;
    let mut structurer = Structurer { cfg, analyses };
    let mut body = structurer.do_tree(structurer.cfg.entry, &[])?;
    drop_redundant_terminal_continues(&mut body);
    Ok(body)
}

#[derive(Debug, Clone)]
pub struct Cfg {
    entry: CfgNodeId,
    nodes: Vec<CfgNode>,
    predecessors: Vec<Vec<CfgNodeId>>,
}

#[derive(Debug, Clone)]
struct CfgNode {
    op: OperationName,
    statements: Vec<Stmt>,
    transfer: Transfer,
    counted_loop: Option<(SsaValue, Expr)>,
}

#[derive(Debug, Clone)]
enum Transfer {
    Goto(CfgNodeId),
    If {
        condition: SsaValue,
        then_target: CfgNodeId,
        else_target: CfgNodeId,
    },
    Switch {
        selector: SsaValue,
        targets: Vec<CfgNodeId>,
    },
    Return,
}

impl Cfg {
    pub fn from_hypergraph(
        f: &OpenHypergraph<Obj, Arr>,
        semantics: &impl ArrowSemantics,
    ) -> Result<Self, StructuredError> {
        let mut consumers: HashMap<NodeId, Vec<CfgNodeId>> = HashMap::new();
        for (edge_index, adjacency) in f.hypergraph.adjacency.iter().enumerate() {
            for source in &adjacency.sources {
                consumers.entry(*source).or_default().push(edge_index);
            }
        }

        let mut entry_edges = Vec::new();
        // One structured program has one entry point. Additional open sources
        // are external state alternatives, not extra CFG entries.
        if let Some(source) = f.sources.first() {
            if let Some(edges) = consumers.get(source) {
                push_unique_all(&mut entry_edges, edges.iter().copied());
            }
        }
        if entry_edges.is_empty() && !f.hypergraph.edges.is_empty() {
            entry_edges.push(0);
        }

        let entry = match entry_edges.as_slice() {
            [edge] => *edge,
            [] => return Err(StructuredError::MissingEntry),
            _ => {
                return Err(StructuredError::UnsupportedEntry {
                    node: "entry".to_string(),
                    successors: entry_edges.len(),
                });
            }
        };

        let graph_targets: HashSet<NodeId> = f.targets.iter().copied().collect();
        let exit_node = (!graph_targets.is_empty()).then_some(f.hypergraph.edges.len());
        let mut nodes = Vec::new();
        let mut arrows = Vec::new();
        for (edge_index, op) in f.hypergraph.edges.iter().enumerate() {
            let op = op.to_string();
            let successors =
                edge_successors(f, edge_index, &consumers, &graph_targets, exit_node, &op);
            let arrow = ArrowInstance {
                id: edge_index,
                op: op.clone(),
                inputs: f.hypergraph.adjacency[edge_index]
                    .sources
                    .iter()
                    .map(|node| wire_name(*node))
                    .collect(),
                outputs: f.hypergraph.adjacency[edge_index]
                    .targets
                    .iter()
                    .map(|node| wire_name(*node))
                    .collect(),
                branch_arity: successors.len(),
            };
            arrows.push(arrow.clone());
            nodes.push(CfgNode {
                counted_loop: semantics.counted_loop(&op),
                op,
                statements: semantics.statements(&arrow),
                transfer: Transfer::Return,
            });
        }

        if !graph_targets.is_empty() {
            nodes.push(CfgNode {
                op: "__exit".to_string(),
                statements: Vec::new(),
                transfer: Transfer::Return,
                counted_loop: None,
            });
        }

        for edge_index in 0..f.hypergraph.edges.len() {
            let arrow = arrows[edge_index].clone();
            let successors = edge_successors(
                f,
                edge_index,
                &consumers,
                &graph_targets,
                (!graph_targets.is_empty()).then_some(f.hypergraph.edges.len()),
                &nodes[edge_index].op,
            );
            nodes[edge_index].transfer =
                transfer_for_successors(&mut nodes, arrow, successors, semantics);
        }

        let mut predecessors = vec![Vec::new(); nodes.len()];
        for (node_index, node) in nodes.iter().enumerate() {
            for successor in node.successors() {
                predecessors[successor].push(node_index);
            }
        }

        Ok(Self {
            entry,
            nodes,
            predecessors,
        })
    }

    fn label(&self, node: CfgNodeId) -> String {
        format!("n{node}")
    }
}

#[derive(Debug, Clone)]
struct Analyses {
    rpo_index: Vec<usize>,
    children: Vec<Vec<CfgNodeId>>,
    merge_nodes: HashSet<CfgNodeId>,
    loop_headers: HashSet<CfgNodeId>,
}

impl Analyses {
    fn new(cfg: &Cfg) -> Result<Self, StructuredError> {
        let rpo = reverse_postorder(cfg);
        let mut rpo_index = vec![usize::MAX; cfg.nodes.len()];
        for (index, node) in rpo.iter().enumerate() {
            rpo_index[*node] = index;
        }

        let dominators = dominators(cfg, &rpo);
        let idom = immediate_dominators(cfg, &dominators);
        let mut children = vec![Vec::new(); cfg.nodes.len()];
        for (node, parent) in idom.iter().enumerate() {
            if let Some(parent) = parent {
                children[*parent].push(node);
            }
        }
        for children in &mut children {
            children.sort_by_key(|node| rpo_index[*node]);
        }

        let mut forward_inedges = vec![0usize; cfg.nodes.len()];
        let mut loop_headers = HashSet::new();
        for (node_index, node) in cfg.nodes.iter().enumerate() {
            for successor in node.successors() {
                if rpo_index[successor] <= rpo_index[node_index] {
                    if !dominators[node_index].contains(&successor) {
                        return Err(StructuredError::IrreducibleBackEdge {
                            from: cfg.label(node_index),
                            to: cfg.label(successor),
                        });
                    }
                    loop_headers.insert(successor);
                } else {
                    forward_inedges[successor] += 1;
                }
            }
        }

        let merge_nodes = forward_inedges
            .iter()
            .enumerate()
            .filter_map(|(node, count)| (*count >= 2).then_some(node))
            .collect();

        Ok(Self {
            rpo_index,
            children,
            merge_nodes,
            loop_headers,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ContextFrame {
    IfThenElse,
    LoopHeadedBy(CfgNodeId),
    BlockFollowedBy(CfgNodeId),
}

struct Structurer {
    cfg: Cfg,
    analyses: Analyses,
}

impl Structurer {
    fn do_tree(
        &mut self,
        node: CfgNodeId,
        context: &[ContextFrame],
    ) -> Result<Vec<Stmt>, StructuredError> {
        if self.is_counted_loop(node) {
            return self.do_counted_loop(node, context);
        }

        let mut inner_context = context.to_vec();
        let mut code = if self.analyses.loop_headers.contains(&node) {
            inner_context.insert(0, ContextFrame::LoopHeadedBy(node));
            vec![Stmt::Loop {
                label: self.cfg.label(node),
                body: self.node_within(node, self.merge_children(node), &inner_context)?,
            }]
        } else {
            self.node_within(node, self.merge_children(node), context)?
        };
        drop_redundant_terminal_continues(&mut code);
        Ok(code)
    }

    fn do_counted_loop(
        &mut self,
        node: CfgNodeId,
        context: &[ContextFrame],
    ) -> Result<Vec<Stmt>, StructuredError> {
        let successors = self.cfg.nodes[node].successors();
        let [body_target, exit_target] = successors.as_slice() else {
            return self.node_within(node, self.merge_children(node), context);
        };

        let mut loop_context = context.to_vec();
        loop_context.insert(0, ContextFrame::LoopHeadedBy(node));

        let (var, extent) = self.cfg.nodes[node]
            .counted_loop
            .clone()
            .expect("is_counted_loop checked counted_loop");
        let mut body = self.do_branch(node, *body_target, &loop_context)?;
        drop_redundant_terminal_continues(&mut body);

        let mut code = vec![Stmt::For {
            label: self.cfg.label(node),
            var,
            extent,
            body,
        }];
        code.extend(self.do_branch(node, *exit_target, context)?);
        Ok(code)
    }

    fn node_within(
        &mut self,
        node: CfgNodeId,
        mut merge_children: Vec<CfgNodeId>,
        context: &[ContextFrame],
    ) -> Result<Vec<Stmt>, StructuredError> {
        if let Some(merge_child) = merge_children.pop() {
            let mut block_context = context.to_vec();
            block_context.insert(0, ContextFrame::BlockFollowedBy(merge_child));
            let mut code = vec![Stmt::Block {
                label: self.cfg.label(merge_child),
                body: self.node_within(node, merge_children, &block_context)?,
            }];
            code.extend(self.do_tree(merge_child, context)?);
            return Ok(code);
        }

        let cfg_node = self.cfg.nodes[node].clone();
        let mut code = cfg_node.statements;
        match cfg_node.transfer {
            Transfer::Return => code.push(Stmt::Return),
            Transfer::Goto(target) => code.extend(self.do_branch(node, target, context)?),
            Transfer::If {
                condition,
                then_target,
                else_target,
            } => {
                let mut then_context = context.to_vec();
                then_context.insert(0, ContextFrame::IfThenElse);
                let else_context = then_context.clone();
                code.push(Stmt::If {
                    condition,
                    then_body: self.do_branch(node, then_target, &then_context)?,
                    else_body: self.do_branch(node, else_target, &else_context)?,
                });
            }
            Transfer::Switch { selector, targets } => {
                let mut case_bodies = Vec::new();
                for target in targets {
                    let mut case_context = context.to_vec();
                    case_context.insert(0, ContextFrame::IfThenElse);
                    case_bodies.push(self.do_branch(node, target, &case_context)?);
                }
                code.push(Stmt::Switch {
                    selector,
                    cases: case_bodies,
                });
            }
        }
        Ok(code)
    }

    fn do_branch(
        &mut self,
        source: CfgNodeId,
        target: CfgNodeId,
        context: &[ContextFrame],
    ) -> Result<Vec<Stmt>, StructuredError> {
        if self.is_backward(source, target) {
            return Ok(vec![Stmt::Continue(self.cfg.label(target))]);
        }
        if self.analyses.merge_nodes.contains(&target) {
            self.index(target, context)?;
            return Ok(vec![Stmt::Break(self.cfg.label(target))]);
        }
        self.do_tree(target, context)
    }

    fn merge_children(&self, node: CfgNodeId) -> Vec<CfgNodeId> {
        let mut children = self.analyses.children[node]
            .iter()
            .copied()
            .filter(|child| self.analyses.merge_nodes.contains(child))
            .collect::<Vec<_>>();
        children.sort_by_key(|child| self.analyses.rpo_index[*child]);
        children
    }

    fn is_backward(&self, source: CfgNodeId, target: CfgNodeId) -> bool {
        self.analyses.rpo_index[target] <= self.analyses.rpo_index[source]
    }

    fn index(&self, target: CfgNodeId, context: &[ContextFrame]) -> Result<usize, StructuredError> {
        for (index, frame) in context.iter().enumerate() {
            let matches = match frame {
                ContextFrame::IfThenElse => false,
                ContextFrame::LoopHeadedBy(label) | ContextFrame::BlockFollowedBy(label) => {
                    *label == target
                }
            };
            if matches {
                return Ok(index);
            }
        }
        Err(StructuredError::MissingContext(self.cfg.label(target)))
    }

    fn is_counted_loop(&self, node: CfgNodeId) -> bool {
        self.analyses.loop_headers.contains(&node)
            && self.cfg.nodes[node].successors().len() == 2
            && self.cfg.nodes[node].counted_loop.is_some()
    }
}

impl CfgNode {
    fn successors(&self) -> Vec<CfgNodeId> {
        match &self.transfer {
            Transfer::Goto(target) => vec![*target],
            Transfer::If {
                then_target,
                else_target,
                ..
            } => vec![*then_target, *else_target],
            Transfer::Switch { targets, .. } => targets.clone(),
            Transfer::Return => Vec::new(),
        }
    }
}

fn edge_successors(
    f: &OpenHypergraph<Obj, Arr>,
    edge_index: CfgNodeId,
    consumers: &HashMap<NodeId, Vec<CfgNodeId>>,
    graph_targets: &HashSet<NodeId>,
    exit_node: Option<CfgNodeId>,
    op: &str,
) -> Vec<CfgNodeId> {
    let mut successors = Vec::new();
    for target in &f.hypergraph.adjacency[edge_index].targets {
        if graph_targets.contains(target) {
            if boundary_target_is_control_successor(op) {
                if let Some(exit_node) = exit_node {
                    push_unique_all(&mut successors, [exit_node]);
                }
            }
            continue;
        }
        if let Some(edges) = consumers.get(target) {
            push_unique_all(&mut successors, edges.iter().copied());
        }
    }
    successors
}

fn transfer_for_successors(
    nodes: &mut Vec<CfgNode>,
    arrow: ArrowInstance,
    successors: Vec<CfgNodeId>,
    semantics: &impl ArrowSemantics,
) -> Transfer {
    match successors.as_slice() {
        [] => Transfer::Return,
        [target] => Transfer::Goto(*target),
        [then_target, else_target] => {
            let condition = branch_condition_value(&arrow, 0);
            let payload = branch_payload(&arrow);
            let then_target = append_binding_node(
                nodes,
                &arrow,
                0,
                branch_binding(&arrow, 0, &payload),
                *then_target,
            );
            let else_target = append_binding_node(
                nodes,
                &arrow,
                1,
                branch_binding(&arrow, 1, &payload),
                *else_target,
            );
            let branch_node = nodes.len();
            nodes.push(CfgNode {
                op: format!("{}.__branch", arrow.op),
                statements: vec![Stmt::Assign {
                    lhs: condition.clone(),
                    rhs: semantics.branch_condition_rhs(&arrow, 0),
                }],
                transfer: Transfer::If {
                    condition,
                    then_target,
                    else_target,
                },
                counted_loop: None,
            });
            Transfer::Goto(branch_node)
        }
        targets => {
            let payload = branch_payload(&arrow);
            let targets = targets
                .iter()
                .enumerate()
                .map(|(index, target)| {
                    append_binding_node(
                        nodes,
                        &arrow,
                        index,
                        branch_binding(&arrow, index, &payload),
                        *target,
                    )
                })
                .collect();
            let branch_node = nodes.len();
            nodes.push(CfgNode {
                op: format!("{}.__branch", arrow.op),
                statements: Vec::new(),
                transfer: Transfer::Switch {
                    selector: semantics.selector(&arrow),
                    targets,
                },
                counted_loop: None,
            });
            Transfer::Goto(branch_node)
        }
    }
}

fn append_binding_node(
    nodes: &mut Vec<CfgNode>,
    arrow: &ArrowInstance,
    output: usize,
    bind: Option<(SsaValue, SsaValue)>,
    target: CfgNodeId,
) -> CfgNodeId {
    let Some((lhs, rhs)) = bind else {
        return target;
    };
    let node = nodes.len();
    nodes.push(CfgNode {
        op: format!("{}.__bind{output}", arrow.op),
        statements: vec![Stmt::Assign { lhs, rhs }],
        transfer: Transfer::Goto(target),
        counted_loop: None,
    });
    node
}

fn branch_payload(arrow: &ArrowInstance) -> SsaValue {
    format!("p{}", arrow.id)
}

fn branch_condition_value(arrow: &ArrowInstance, output: usize) -> SsaValue {
    format!("c{}_{}", arrow.id, output)
}

fn branch_binding(
    arrow: &ArrowInstance,
    output: usize,
    payload: &str,
) -> Option<(SsaValue, SsaValue)> {
    arrow
        .outputs
        .get(output)
        .map(|wire| (wire.clone(), payload.to_string()))
}

fn boundary_target_is_control_successor(op: &str) -> bool {
    // `gpu.sync` exposes phase-token outputs at the graph boundary, but those
    // are data/control products rather than alternate control exits.
    op != "gpu.sync"
}

fn wire_name(node: NodeId) -> SsaValue {
    format!("w{}", node.0)
}

fn reverse_postorder(cfg: &Cfg) -> Vec<CfgNodeId> {
    fn visit(cfg: &Cfg, node: CfgNodeId, seen: &mut [bool], postorder: &mut Vec<CfgNodeId>) {
        if seen[node] {
            return;
        }
        seen[node] = true;
        for successor in cfg.nodes[node].successors() {
            visit(cfg, successor, seen, postorder);
        }
        postorder.push(node);
    }

    let mut seen = vec![false; cfg.nodes.len()];
    let mut postorder = Vec::new();
    visit(cfg, cfg.entry, &mut seen, &mut postorder);
    postorder.reverse();
    postorder
}

fn dominators(cfg: &Cfg, rpo: &[CfgNodeId]) -> Vec<BTreeSet<CfgNodeId>> {
    let all_reachable = rpo.iter().copied().collect::<BTreeSet<_>>();
    let mut doms = vec![BTreeSet::new(); cfg.nodes.len()];
    for node in rpo {
        doms[*node] = all_reachable.clone();
    }
    doms[cfg.entry] = BTreeSet::from([cfg.entry]);

    let mut changed = true;
    while changed {
        changed = false;
        for node in rpo.iter().copied().filter(|node| *node != cfg.entry) {
            let reachable_preds = cfg.predecessors[node]
                .iter()
                .copied()
                .filter(|pred| !doms[*pred].is_empty())
                .collect::<Vec<_>>();
            let mut new_doms = if let Some((first, rest)) = reachable_preds.split_first() {
                let mut intersection = doms[*first].clone();
                for pred in rest {
                    intersection = intersection
                        .intersection(&doms[*pred])
                        .copied()
                        .collect::<BTreeSet<_>>();
                }
                intersection
            } else {
                BTreeSet::new()
            };
            new_doms.insert(node);
            if new_doms != doms[node] {
                doms[node] = new_doms;
                changed = true;
            }
        }
    }

    doms
}

fn immediate_dominators(cfg: &Cfg, doms: &[BTreeSet<CfgNodeId>]) -> Vec<Option<CfgNodeId>> {
    let mut idom = vec![None; cfg.nodes.len()];
    for node in 0..cfg.nodes.len() {
        if node == cfg.entry || doms[node].is_empty() {
            continue;
        }
        let strict = doms[node]
            .iter()
            .copied()
            .filter(|dom| *dom != node)
            .collect::<Vec<_>>();
        idom[node] = strict.iter().copied().find(|candidate| {
            strict
                .iter()
                .all(|other| candidate == other || doms[*candidate].contains(other))
        });
    }
    idom
}

fn push_unique_all(target: &mut Vec<CfgNodeId>, values: impl IntoIterator<Item = CfgNodeId>) {
    for value in values {
        if !target.contains(&value) {
            target.push(value);
        }
    }
}

fn drop_redundant_terminal_continues(stmts: &mut Vec<Stmt>) {
    for stmt in stmts.iter_mut() {
        match stmt {
            Stmt::Block { body, .. } | Stmt::Loop { body, .. } | Stmt::For { body, .. } => {
                drop_redundant_terminal_continues(body)
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                drop_redundant_terminal_continues(then_body);
                drop_redundant_terminal_continues(else_body);
            }
            Stmt::Switch { cases, .. } => {
                for body in cases {
                    drop_redundant_terminal_continues(body);
                }
            }
            _ => {}
        }
    }
    if matches!(stmts.last(), Some(Stmt::Continue(_))) {
        stmts.pop();
    }
}

fn sanitize_ident(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}
