use super::ir::Stmt;
use crate::lang::{Arr, Obj};
use open_hypergraphs::lax::{EdgeId, NodeId, OpenHypergraph};
use std::collections::{BTreeSet, HashMap, HashSet};

pub trait ArrowSemantics {
    fn actions(&self, op: &str) -> Vec<Stmt>;

    fn condition(&self, op: &str) -> String {
        format!("/* {} */ 1", sanitize_ident(op))
    }

    fn selector(&self, op: &str) -> String {
        format!("/* {} */ 0", sanitize_ident(op))
    }

    fn counted_loop(&self, _op: &str) -> Option<(String, String)> {
        None
    }
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

pub fn structure(cfg: Cfg, semantics: impl ArrowSemantics) -> Result<Vec<Stmt>, StructuredError> {
    let analyses = Analyses::new(&cfg)?;
    let mut structurer = Structurer {
        cfg,
        analyses,
        semantics,
    };
    let mut body = structurer.do_tree(structurer.cfg.entry, &[])?;
    drop_redundant_terminal_continues(&mut body);
    Ok(body)
}

#[derive(Debug, Clone)]
pub struct Cfg {
    entry: usize,
    nodes: Vec<CfgNode>,
    predecessors: Vec<Vec<usize>>,
}

#[derive(Debug, Clone)]
struct CfgNode {
    edge: EdgeId,
    op: String,
    successors: Vec<usize>,
}

impl Cfg {
    pub fn from_hypergraph(f: &OpenHypergraph<Obj, Arr>) -> Result<Self, StructuredError> {
        let mut consumers: HashMap<NodeId, Vec<usize>> = HashMap::new();
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
        let mut nodes = Vec::new();
        for (edge_index, op) in f.hypergraph.edges.iter().enumerate() {
            let mut successors = Vec::new();
            for target in &f.hypergraph.adjacency[edge_index].targets {
                if graph_targets.contains(target) {
                    continue;
                }
                if let Some(edges) = consumers.get(target) {
                    push_unique_all(&mut successors, edges.iter().copied());
                }
            }
            nodes.push(CfgNode {
                edge: EdgeId(edge_index),
                op: op.to_string(),
                successors,
            });
        }

        let mut predecessors = vec![Vec::new(); nodes.len()];
        for node in &nodes {
            for successor in &node.successors {
                predecessors[*successor].push(node.edge.0);
            }
        }

        Ok(Self {
            entry,
            nodes,
            predecessors,
        })
    }

    fn label(&self, node: usize) -> String {
        format!("n{node}")
    }
}

#[derive(Debug, Clone)]
struct Analyses {
    rpo_index: Vec<usize>,
    children: Vec<Vec<usize>>,
    merge_nodes: HashSet<usize>,
    loop_headers: HashSet<usize>,
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
        for node in &cfg.nodes {
            for successor in &node.successors {
                if rpo_index[*successor] <= rpo_index[node.edge.0] {
                    if !dominators[node.edge.0].contains(successor) {
                        return Err(StructuredError::IrreducibleBackEdge {
                            from: cfg.label(node.edge.0),
                            to: cfg.label(*successor),
                        });
                    }
                    loop_headers.insert(*successor);
                } else {
                    forward_inedges[*successor] += 1;
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
    LoopHeadedBy(usize),
    BlockFollowedBy(usize),
}

struct Structurer<S> {
    cfg: Cfg,
    analyses: Analyses,
    semantics: S,
}

impl<S: ArrowSemantics> Structurer<S> {
    fn do_tree(
        &mut self,
        node: usize,
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
        node: usize,
        context: &[ContextFrame],
    ) -> Result<Vec<Stmt>, StructuredError> {
        let successors = self.cfg.nodes[node].successors.clone();
        let [body_target, exit_target] = successors.as_slice() else {
            return self.node_within(node, self.merge_children(node), context);
        };

        let mut loop_context = context.to_vec();
        loop_context.insert(0, ContextFrame::LoopHeadedBy(node));

        let (var, extent) = self
            .semantics
            .counted_loop(&self.cfg.nodes[node].op)
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
        node: usize,
        mut merge_children: Vec<usize>,
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
        let mut code = self.semantics.actions(&cfg_node.op);
        match cfg_node.successors.as_slice() {
            [] => code.push(Stmt::Return),
            [target] => code.extend(self.do_branch(node, *target, context)?),
            [then_target, else_target] => {
                let mut then_context = context.to_vec();
                then_context.insert(0, ContextFrame::IfThenElse);
                let else_context = then_context.clone();
                code.push(Stmt::If {
                    condition: self.semantics.condition(&cfg_node.op),
                    then_body: self.do_branch(node, *then_target, &then_context)?,
                    else_body: self.do_branch(node, *else_target, &else_context)?,
                });
            }
            successors => {
                let mut cases = Vec::new();
                for successor in successors {
                    let mut case_context = context.to_vec();
                    case_context.insert(0, ContextFrame::IfThenElse);
                    cases.push(self.do_branch(node, *successor, &case_context)?);
                }
                code.push(Stmt::Switch {
                    selector: self.semantics.selector(&cfg_node.op),
                    cases,
                });
            }
        }
        Ok(code)
    }

    fn do_branch(
        &mut self,
        source: usize,
        target: usize,
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

    fn merge_children(&self, node: usize) -> Vec<usize> {
        let mut children = self.analyses.children[node]
            .iter()
            .copied()
            .filter(|child| self.analyses.merge_nodes.contains(child))
            .collect::<Vec<_>>();
        children.sort_by_key(|child| self.analyses.rpo_index[*child]);
        children
    }

    fn is_backward(&self, source: usize, target: usize) -> bool {
        self.analyses.rpo_index[target] <= self.analyses.rpo_index[source]
    }

    fn index(&self, target: usize, context: &[ContextFrame]) -> Result<usize, StructuredError> {
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

    fn is_counted_loop(&self, node: usize) -> bool {
        self.analyses.loop_headers.contains(&node)
            && self.cfg.nodes[node].successors.len() == 2
            && self
                .semantics
                .counted_loop(&self.cfg.nodes[node].op)
                .is_some()
    }
}

fn reverse_postorder(cfg: &Cfg) -> Vec<usize> {
    fn visit(cfg: &Cfg, node: usize, seen: &mut [bool], postorder: &mut Vec<usize>) {
        if seen[node] {
            return;
        }
        seen[node] = true;
        for successor in &cfg.nodes[node].successors {
            visit(cfg, *successor, seen, postorder);
        }
        postorder.push(node);
    }

    let mut seen = vec![false; cfg.nodes.len()];
    let mut postorder = Vec::new();
    visit(cfg, cfg.entry, &mut seen, &mut postorder);
    postorder.reverse();
    postorder
}

fn dominators(cfg: &Cfg, rpo: &[usize]) -> Vec<BTreeSet<usize>> {
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

fn immediate_dominators(cfg: &Cfg, doms: &[BTreeSet<usize>]) -> Vec<Option<usize>> {
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

fn push_unique_all(target: &mut Vec<usize>, values: impl IntoIterator<Item = usize>) {
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
