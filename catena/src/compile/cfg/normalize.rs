use std::collections::{HashMap, HashSet};

use super::{
    model::{Cfg, CfgEdge, CfgNode, CfgNodeId, Transfer},
    wiring::{predecessors, remap_transfer_targets},
};

pub(super) fn normalize_cfg(mut cfg: Cfg) -> Cfg {
    for node in &mut cfg.nodes {
        prune_unused_params(node);
    }
    bypass_empty_goto_nodes(&mut cfg.nodes, &mut cfg.entry);
    compact_node_ids(&mut cfg.nodes, &mut cfg.entry);
    cfg.nodes.sort_by_key(|node| node.id);
    cfg.predecessors = predecessors(&cfg.nodes);
    cfg
}

fn prune_unused_params(node: &mut CfgNode) {
    let mut used = node
        .block
        .iter()
        .flat_map(|instruction| instruction.args.iter().copied())
        .collect::<HashSet<_>>();
    match &node.transfer {
        Transfer::Goto(edge) => {
            used.extend(edge.args.iter().copied());
        }
        Transfer::If {
            condition,
            then_edge,
            else_edge,
        } => {
            used.insert(*condition);
            used.extend(then_edge.args.iter().copied());
            used.extend(else_edge.args.iter().copied());
        }
        Transfer::Return(values) => {
            used.extend(values.iter().copied());
        }
    }
    node.params.retain(|param| used.contains(param));
}

fn bypass_empty_goto_nodes(nodes: &mut Vec<CfgNode>, entry: &mut CfgNodeId) {
    let bypasses = nodes
        .iter()
        .filter_map(|node| {
            if !node.params.is_empty() || !node.block.is_empty() {
                return None;
            }
            let Transfer::Goto(edge) = &node.transfer else {
                return None;
            };
            Some((node.id, edge.clone()))
        })
        .collect::<HashMap<_, _>>();
    if bypasses.is_empty() {
        return;
    }

    while let Some(edge) = bypasses.get(entry) {
        *entry = edge.target;
    }
    for node in nodes.iter_mut() {
        node.transfer = bypass_transfer(node.transfer.clone(), &bypasses);
    }
    nodes.retain(|node| !bypasses.contains_key(&node.id));
}

fn bypass_transfer(transfer: Transfer, bypasses: &HashMap<CfgNodeId, CfgEdge>) -> Transfer {
    match transfer {
        Transfer::Goto(edge) => Transfer::Goto(bypass_edge(edge, bypasses)),
        Transfer::If {
            condition,
            then_edge,
            else_edge,
        } => Transfer::If {
            condition,
            then_edge: bypass_edge(then_edge, bypasses),
            else_edge: bypass_edge(else_edge, bypasses),
        },
        Transfer::Return(values) => Transfer::Return(values),
    }
}

fn bypass_edge(mut edge: CfgEdge, bypasses: &HashMap<CfgNodeId, CfgEdge>) -> CfgEdge {
    while let Some(next) = bypasses.get(&edge.target) {
        edge.target = next.target;
        edge.args = next.args.clone();
    }
    edge
}

fn compact_node_ids(nodes: &mut [CfgNode], entry: &mut CfgNodeId) {
    nodes.sort_by_key(|node| node.id);
    let node_id_by_old = nodes
        .iter()
        .enumerate()
        .map(|(new, node)| (node.id, new))
        .collect::<HashMap<_, _>>();
    if let Some(new_entry) = node_id_by_old.get(entry).copied() {
        *entry = new_entry;
    }
    for node in nodes {
        node.id = node_id_by_old[&node.id];
        node.transfer = remap_transfer_targets(node.transfer.clone(), &node_id_by_old);
    }
}
