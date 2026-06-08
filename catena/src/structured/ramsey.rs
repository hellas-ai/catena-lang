use super::ir::{Primitive, Stmt};
use crate::compile::cfg::{BlockInstruction, Cfg, CfgEdge, CfgNodeId, Transfer};
use std::collections::{BTreeSet, HashSet};

#[derive(Debug, thiserror::Error)]
pub enum RamseyError {
    #[error("control-flow graph has an irreducible back edge from {from} to {to}")]
    IrreducibleBackEdge { from: String, to: String },
    #[error("branch target {0} is not in the structured context")]
    MissingContext(String),
}

pub fn structure(
    cfg: Cfg,
    variable_name: impl Fn(crate::compile::cfg::VariableId) -> String + 'static,
) -> Result<Vec<Stmt>, RamseyError> {
    let analyses = Analyses::new(&cfg)?;
    let mut structurer = Structurer {
        cfg,
        analyses,
        variable_name: Box::new(variable_name),
    };
    let mut body = structurer.do_tree(structurer.cfg.entry, &[])?;
    drop_redundant_terminal_continues(&mut body);
    simplify_redundant_blocks(&mut body);
    Ok(body)
}

#[derive(Debug, Clone)]
struct Analyses {
    rpo_index: Vec<usize>,
    children: Vec<Vec<CfgNodeId>>,
    merge_nodes: HashSet<CfgNodeId>,
    loop_headers: HashSet<CfgNodeId>,
}

impl Analyses {
    fn new(cfg: &Cfg) -> Result<Self, RamseyError> {
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
                        return Err(RamseyError::IrreducibleBackEdge {
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
    variable_name: Box<dyn Fn(crate::compile::cfg::VariableId) -> String>,
}

impl Structurer {
    fn do_tree(
        &mut self,
        node: CfgNodeId,
        context: &[ContextFrame],
    ) -> Result<Vec<Stmt>, RamseyError> {
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

    fn node_within(
        &mut self,
        node: CfgNodeId,
        mut merge_children: Vec<CfgNodeId>,
        context: &[ContextFrame],
    ) -> Result<Vec<Stmt>, RamseyError> {
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
        let mut code = self.block_statements(&cfg_node.block);
        match cfg_node.transfer {
            Transfer::Return(values) => {
                code.push(Stmt::Return(
                    values
                        .into_iter()
                        .map(|id| (self.variable_name)(id))
                        .collect(),
                ));
            }
            Transfer::Goto(edge) => code.extend(self.do_edge(node, &edge, context)?),
            Transfer::If {
                condition,
                then_edge,
                else_edge,
            } => {
                let mut then_context = context.to_vec();
                then_context.insert(0, ContextFrame::IfThenElse);
                let else_context = then_context.clone();
                code.push(Stmt::If {
                    condition: (self.variable_name)(condition),
                    then_body: self.do_edge(node, &then_edge, &then_context)?,
                    else_body: self.do_edge(node, &else_edge, &else_context)?,
                });
            }
        }
        Ok(code)
    }

    fn do_edge(
        &mut self,
        source: CfgNodeId,
        edge: &CfgEdge,
        context: &[ContextFrame],
    ) -> Result<Vec<Stmt>, RamseyError> {
        let mut code = self.edge_bindings(edge);
        code.extend(self.do_branch(source, edge.target, context)?);
        Ok(code)
    }

    fn do_branch(
        &mut self,
        source: CfgNodeId,
        target: CfgNodeId,
        context: &[ContextFrame],
    ) -> Result<Vec<Stmt>, RamseyError> {
        if self.is_backward(source, target) {
            return Ok(vec![Stmt::Continue(self.cfg.label(target))]);
        }
        if self.analyses.merge_nodes.contains(&target) {
            self.index(target, context)?;
            return Ok(vec![Stmt::Break(self.cfg.label(target))]);
        }
        self.do_tree(target, context)
    }

    fn edge_bindings(&self, edge: &CfgEdge) -> Vec<Stmt> {
        let target = &self.cfg.nodes[edge.target];
        target
            .params
            .iter()
            .zip(edge.args.iter())
            .filter(|(param, arg)| param != arg)
            .map(|(param, arg)| Stmt::Assign {
                lhs: (self.variable_name)(*param),
                rhs: (self.variable_name)(*arg),
            })
            .collect()
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

    fn index(&self, target: CfgNodeId, context: &[ContextFrame]) -> Result<usize, RamseyError> {
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
        Err(RamseyError::MissingContext(self.cfg.label(target)))
    }

    fn block_statements(&self, block: &[BlockInstruction]) -> Vec<Stmt> {
        block
            .iter()
            .map(|instruction| self.block_instruction_statement(instruction))
            .collect()
    }

    fn block_instruction_statement(&self, instruction: &BlockInstruction) -> Stmt {
        let outputs = instruction
            .results
            .iter()
            .map(|id| (self.variable_name)(*id))
            .collect::<Vec<_>>();
        if instruction.operation == "gpu.sync" {
            return Stmt::Barrier;
        }
        Stmt::Primitive(Primitive {
            name: instruction.operation.clone(),
            inputs: instruction
                .args
                .iter()
                .map(|id| (self.variable_name)(*id))
                .collect(),
            outputs,
            code: String::new(),
        })
    }
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

fn simplify_redundant_blocks(stmts: &mut Vec<Stmt>) {
    let mut simplified = Vec::new();
    for mut stmt in std::mem::take(stmts) {
        simplify_stmt(&mut stmt);
        match stmt {
            Stmt::Block { label, mut body } => {
                remove_fallthrough_breaks_to(&mut body, &label);
                if !contains_break_to(&body, &label) {
                    simplified.extend(body);
                } else {
                    simplified.push(Stmt::Block { label, body });
                }
            }
            other => simplified.push(other),
        }
    }
    *stmts = simplified;
}

fn simplify_stmt(stmt: &mut Stmt) {
    match stmt {
        Stmt::Block { body, .. } | Stmt::Loop { body, .. } | Stmt::For { body, .. } => {
            simplify_redundant_blocks(body)
        }
        Stmt::If {
            then_body,
            else_body,
            ..
        } => {
            simplify_redundant_blocks(then_body);
            simplify_redundant_blocks(else_body);
        }
        Stmt::Switch { cases, .. } => {
            for body in cases {
                simplify_redundant_blocks(body);
            }
        }
        Stmt::Break(_)
        | Stmt::Continue(_)
        | Stmt::Return(_)
        | Stmt::Barrier
        | Stmt::Assign { .. }
        | Stmt::Call { .. }
        | Stmt::Primitive(_)
        | Stmt::Comment(_) => {}
    }
}

fn remove_fallthrough_breaks_to(stmts: &mut Vec<Stmt>, label: &str) {
    remove_terminal_break_to(stmts, label);
    for stmt in stmts {
        match stmt {
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                // In a branch body, a terminal `break label` is equivalent to
                // falling out of the enclosing block.
                remove_fallthrough_breaks_to(then_body, label);
                remove_fallthrough_breaks_to(else_body, label);
            }
            Stmt::Switch { cases, .. } => {
                for body in cases {
                    // Each case has its own terminal position.
                    remove_fallthrough_breaks_to(body, label);
                }
            }
            Stmt::Block { body, .. } | Stmt::Loop { body, .. } | Stmt::For { body, .. } => {
                // Keep simplifying nested bodies, but the outer block is only
                // removed after we prove no `break label` remains anywhere.
                remove_fallthrough_breaks_to(body, label);
            }
            Stmt::Break(_)
            | Stmt::Continue(_)
            | Stmt::Return(_)
            | Stmt::Barrier
            | Stmt::Assign { .. }
            | Stmt::Call { .. }
            | Stmt::Primitive(_)
            | Stmt::Comment(_) => {}
        }
    }
}

fn remove_terminal_break_to(stmts: &mut Vec<Stmt>, label: &str) {
    if matches!(stmts.last(), Some(Stmt::Break(break_label)) if break_label == label) {
        stmts.pop();
    }
}

fn contains_break_to(stmts: &[Stmt], label: &str) -> bool {
    stmts.iter().any(|stmt| match stmt {
        Stmt::Block { body, .. } | Stmt::Loop { body, .. } | Stmt::For { body, .. } => {
            contains_break_to(body, label)
        }
        Stmt::If {
            then_body,
            else_body,
            ..
        } => contains_break_to(then_body, label) || contains_break_to(else_body, label),
        Stmt::Switch { cases, .. } => cases.iter().any(|body| contains_break_to(body, label)),
        Stmt::Break(break_label) => break_label == label,
        Stmt::Continue(_)
        | Stmt::Return(_)
        | Stmt::Barrier
        | Stmt::Assign { .. }
        | Stmt::Call { .. }
        | Stmt::Primitive(_)
        | Stmt::Comment(_) => false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_block_when_all_breaks_are_fallthrough() {
        let mut stmts = vec![Stmt::Block {
            label: "n7".to_string(),
            body: vec![Stmt::If {
                condition: "c".to_string(),
                then_body: vec![Stmt::Break("n7".to_string())],
                else_body: vec![Stmt::Break("n7".to_string())],
            }],
        }];

        simplify_redundant_blocks(&mut stmts);

        assert_eq!(
            stmts,
            vec![Stmt::If {
                condition: "c".to_string(),
                then_body: Vec::new(),
                else_body: Vec::new(),
            }]
        );
    }

    #[test]
    fn keeps_block_when_break_to_label_remains() {
        let mut stmts = vec![Stmt::Block {
            label: "n7".to_string(),
            body: vec![Stmt::If {
                condition: "c".to_string(),
                then_body: vec![Stmt::Break("n7".to_string()), Stmt::Return(Vec::new())],
                else_body: vec![Stmt::Break("n7".to_string())],
            }],
        }];

        simplify_redundant_blocks(&mut stmts);

        assert!(matches!(stmts.as_slice(), [Stmt::Block { label, .. }] if label == "n7"));
    }
}
