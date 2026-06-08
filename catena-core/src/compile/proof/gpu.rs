use crate::compile::CompileGraph;

use super::{ProofProperty, ProofRequirement};

const MEMORY_SAFETY: &str = "gpu memory safety";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MemorySafety;

impl ProofProperty for MemorySafety {
    fn requirements(&self, graph: &CompileGraph) -> Vec<ProofRequirement> {
        let mut requirements = Vec::new();
        collect_memory_safety_requirements(graph, &graph.definition_name, &mut requirements);
        requirements
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MemoryAccess {
    op: String,
    memory: String,
    view: String,
}

impl MemoryAccess {
    fn to_requirement(&self, entry: &str) -> ProofRequirement {
        ProofRequirement::new(MEMORY_SAFETY, self.subject(), self.proof_name(entry))
    }

    fn subject(&self) -> String {
        format!("{} view {} for memory {}", self.op, self.view, self.memory)
    }

    fn proof_name(&self, entry: &str) -> String {
        format!("{entry}.{}.{}.safe-access", self.memory, self.view)
    }
}

fn collect_memory_safety_requirements(
    graph: &CompileGraph,
    entry: &str,
    requirements: &mut Vec<ProofRequirement>,
) {
    let accesses = find_memory_accesses(graph);
    requirements.extend(build_memory_safety_requirements(entry, accesses));

    for child in &graph.children {
        collect_memory_safety_requirements(&child.graph, entry, requirements);
    }
}

fn build_memory_safety_requirements(
    entry: &str,
    accesses: Vec<MemoryAccess>,
) -> impl Iterator<Item = ProofRequirement> + '_ {
    accesses
        .into_iter()
        .map(|access| access.to_requirement(entry))
}

fn find_memory_accesses(graph: &CompileGraph) -> Vec<MemoryAccess> {
    graph
        .graph
        .h
        .x
        .0
        .iter()
        .zip(graph.graph.h.s.clone())
        .filter_map(|(op, sources)| {
            let op = op.to_string();
            memory_access_from_edge(graph, op, &sources.table)
        })
        .collect()
}

fn memory_access_from_edge(
    graph: &CompileGraph,
    op: String,
    sources: &[usize],
) -> Option<MemoryAccess> {
    if !is_gpu_memory_access(&op) {
        return None;
    }

    Some(MemoryAccess {
        op,
        memory: source_name(graph, sources, 0),
        view: source_name(graph, sources, 1),
    })
}

fn source_name(graph: &CompileGraph, sources: &[usize], index: usize) -> String {
    sources
        .get(index)
        .and_then(|node| graph.source_variable_names.get(node))
        .cloned()
        .unwrap_or_else(|| "<unknown>".to_string())
}

fn is_gpu_memory_access(name: &str) -> bool {
    matches!(
        name,
        "gpu.global.load" | "gpu.global.store" | "gpu.shared.load" | "gpu.shared.store"
    )
}
