use std::collections::BTreeMap;

use hexpr::Operation;

use crate::{
    codegen::{
        GpuValue, GpuVar,
        fn_ptrs::FnPtrSymbol,
        lower_types::{CType, LowerTypeError, LoweredType, lower_type},
    },
    report::AnnotatedTerm,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SpecializationKey {
    pub sources: Vec<CType>,
    pub targets: Vec<CType>,
    pub static_inputs: Vec<FnPtrSymbol>,
}

pub struct PendingInstance {
    pub op: Operation,
    pub name: String,
    pub overrides: BTreeMap<usize, LoweredType>,
}

pub fn entrypoint_key(term: &AnnotatedTerm) -> Result<Option<SpecializationKey>, LowerTypeError> {
    let mut sources = Vec::new();
    let mut targets = Vec::new();
    for source in &term.sources {
        if let LoweredType::Runtime(ty) = lower_type(&term.hypergraph.nodes[source.0])? {
            sources.push(ty);
        }
    }
    for target in &term.targets {
        if let LoweredType::Runtime(ty) = lower_type(&term.hypergraph.nodes[target.0])? {
            targets.push(ty);
        }
    }
    if sources.is_empty() && targets.is_empty() {
        return Ok(None);
    }
    Ok(Some(SpecializationKey {
        sources,
        targets,
        static_inputs: Vec::new(),
    }))
}

pub fn specialization_key(inputs: &[GpuValue], outputs: &[GpuVar]) -> Option<SpecializationKey> {
    let mut sources = Vec::new();
    let mut static_inputs = Vec::new();
    for input in inputs {
        match input {
            GpuValue::Var(var) => {
                if let LoweredType::Runtime(ty) = &var.lowered {
                    sources.push(ty.clone());
                }
            }
            GpuValue::FnSymbol(symbol) => static_inputs.push(symbol.clone()),
        }
    }
    let mut targets = Vec::new();
    for output in outputs {
        if let LoweredType::Runtime(ty) = &output.lowered {
            targets.push(ty.clone());
        }
    }
    if sources.is_empty() && targets.is_empty() && static_inputs.is_empty() {
        return None;
    }
    Some(SpecializationKey {
        sources,
        targets,
        static_inputs,
    })
}

pub fn specialization_overrides(
    term: &AnnotatedTerm,
    inputs: &[GpuValue],
    outputs: &[GpuVar],
) -> BTreeMap<usize, LoweredType> {
    let mut overrides = BTreeMap::new();
    for (node, input) in term.sources.iter().zip(inputs.iter()) {
        if let GpuValue::Var(var) = input {
            overrides.insert(node.0, var.lowered.clone());
        }
    }
    for (node, output) in term.targets.iter().zip(outputs.iter()) {
        overrides.insert(node.0, output.lowered.clone());
    }
    overrides
}

#[cfg(test)]
mod tests {
    use super::*;

    use metacat::tree::Tree;
    use open_hypergraphs::lax::{NodeId, OpenHypergraph};

    fn op(name: &str) -> Operation {
        name.parse().unwrap()
    }

    fn node(name: &str, children: Vec<Tree<(), Operation>>) -> Tree<(), Operation> {
        Tree::Node(op(name), 0, children)
    }

    fn var(node: usize, name: &str, lowered: LoweredType) -> GpuVar {
        GpuVar {
            node: NodeId(node),
            name: name.to_string(),
            lowered,
        }
    }

    #[test]
    fn function_symbols_are_part_of_specialization_key() {
        let output = var(2, "x2", LoweredType::Runtime(CType::Bool));
        let foo_key = specialization_key(
            &[GpuValue::FnSymbol(FnPtrSymbol { target: op("foo") })],
            std::slice::from_ref(&output),
        )
        .unwrap();
        let bar_key = specialization_key(
            &[GpuValue::FnSymbol(FnPtrSymbol { target: op("bar") })],
            &[output],
        )
        .unwrap();

        assert_ne!(foo_key, bar_key);
    }

    #[test]
    fn erased_only_generic_definition_is_not_an_entrypoint() {
        let term = OpenHypergraph::identity(vec![Tree::Leaf(0, ())]);

        assert!(entrypoint_key(&term).unwrap().is_none());
    }

    #[test]
    fn runtime_interface_definition_is_an_entrypoint() {
        let term = OpenHypergraph::identity(vec![node("val", vec![node("bool", vec![])])]);

        assert_eq!(
            entrypoint_key(&term).unwrap().unwrap(),
            SpecializationKey {
                sources: vec![CType::Bool],
                targets: vec![CType::Bool],
                static_inputs: Vec::new(),
            }
        );
    }
}
