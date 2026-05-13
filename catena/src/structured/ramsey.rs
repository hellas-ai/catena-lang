use super::{
    cfg::{Cfg, CfgNodeId, StructuredError, Transfer},
    ir::Stmt,
};
use std::collections::{BTreeSet, HashSet};

pub fn structure(cfg: Cfg) -> Result<Vec<Stmt>, StructuredError> {
    let analyses = Analyses::new(&cfg)?;
    let mut structurer = Structurer { cfg, analyses };
    let mut body = structurer.do_tree(structurer.cfg.entry, &[])?;
    drop_redundant_terminal_continues(&mut body);
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
