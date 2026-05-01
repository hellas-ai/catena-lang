use std::collections::HashMap;

use hexpr::Operation;
use metacat::{
    syntax::{Declaration, TheoryBundle},
    theory::OperationKey,
};
use open_hypergraphs::category::Arrow;
use open_hypergraphs::lax::OpenHypergraph;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LiftError {
    #[error("cannot lift {prefix}: missing object constructor `{object}` in target theory")]
    MissingObject {
        prefix: &'static str,
        object: String,
    },
    #[error(
        "cannot lift {prefix}: object constructor `{object}` has profile {source_arity} -> {target_arity}, expected {expected_source_arity} -> {expected_target_arity}"
    )]
    InvalidObjectProfile {
        prefix: &'static str,
        object: String,
        source_arity: usize,
        target_arity: usize,
        expected_source_arity: usize,
        expected_target_arity: usize,
    },
    #[error("cannot lift {prefix}: invalid lifted operation name `{name}`")]
    InvalidLiftedOperationName { prefix: &'static str, name: String },
    #[error("cannot lift {prefix}: failed to add lifted operation `{name}`: {error}")]
    AddOperation {
        prefix: &'static str,
        name: String,
        error: metacat::theory::Error,
    },
}

pub fn lift_data_to_control(
    data: &TheoryBundle,
    control: &TheoryBundle,
) -> Result<TheoryBundle, LiftError> {
    lift_with_tensor(data, control, "data", "product", "unit")
}

pub fn lift_control_to_data(
    control: &TheoryBundle,
    data: &TheoryBundle,
) -> Result<TheoryBundle, LiftError> {
    lift_with_tensor(control, data, "control", "coproduct", "unit")
}

fn lift_with_tensor(
    source: &TheoryBundle,
    target: &TheoryBundle,
    prefix: &'static str,
    tensor: &str,
    unit: &str,
) -> Result<TheoryBundle, LiftError> {
    let tensor_key = require_object(target, prefix, tensor, 2, 1)?;
    let unit_key = require_object(target, prefix, unit, 0, 1)?;
    let mut bundle = clone_bundle(target);

    let mut operations: Vec<_> = source.arrow_theory.operations().collect();
    operations.sort_by_key(|op| op.to_string());

    for op in operations {
        let original_name = op.to_string();
        let lifted_name = format!("{prefix}.{original_name}");
        let lifted_operation: Operation =
            lifted_name
                .parse()
                .map_err(|_| LiftError::InvalidLiftedOperationName {
                    prefix,
                    name: lifted_name.clone(),
                })?;

        let (source_map, target_map) = source.arrow_theory.type_maps(op);
        let lifted_source = lift_object_map(source_map, target, &tensor_key, &unit_key, prefix)?;
        let lifted_target = lift_object_map(target_map, target, &tensor_key, &unit_key, prefix)?;

        bundle
            .arrow_theory
            .add_operation(lifted_operation, lifted_source, lifted_target)
            .map_err(|error| LiftError::AddOperation {
                prefix,
                name: lifted_name.clone(),
                error,
            })?;
    }

    Ok(bundle)
}

fn require_object(
    bundle: &TheoryBundle,
    prefix: &'static str,
    object: &str,
    expected_source_arity: usize,
    expected_target_arity: usize,
) -> Result<OperationKey, LiftError> {
    let key = bundle
        .object_theory
        .get_operation_key(object)
        .ok_or_else(|| LiftError::MissingObject {
            prefix,
            object: object.to_string(),
        })?;
    let (source, target) = bundle.object_theory.type_maps(&key);
    let source_arity = source.target().len();
    let target_arity = target.target().len();
    if source_arity == expected_source_arity && target_arity == expected_target_arity {
        Ok(key)
    } else {
        Err(LiftError::InvalidObjectProfile {
            prefix,
            object: object.to_string(),
            source_arity,
            target_arity,
            expected_source_arity,
            expected_target_arity,
        })
    }
}

fn lift_object_map(
    map: &OpenHypergraph<(), OperationKey>,
    target: &TheoryBundle,
    tensor_key: &OperationKey,
    unit_key: &OperationKey,
    prefix: &'static str,
) -> Result<OpenHypergraph<(), OperationKey>, LiftError> {
    let remapped_edges = map
        .hypergraph
        .edges
        .iter()
        .map(|op| {
            let object = op.to_string();
            target
                .object_theory
                .get_operation_key(&object)
                .ok_or_else(|| LiftError::MissingObject { prefix, object })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut lifted = map
        .clone()
        .with_edges(|_| remapped_edges)
        .expect("edge remapping preserves edge count");

    match lifted.targets.len() {
        0 => {
            let unit_node = lifted.new_node(());
            lifted.new_edge(unit_key.clone(), (Vec::new(), vec![unit_node]));
            lifted.targets = vec![unit_node];
        }
        1 => {}
        _ => {
            let mut inputs = lifted.targets.clone();
            while inputs.len() > 1 {
                let left = inputs.remove(0);
                let right = inputs.remove(0);
                let product_node = lifted.new_node(());
                lifted.new_edge(tensor_key.clone(), (vec![left, right], vec![product_node]));
                inputs.insert(0, product_node);
            }
            lifted.targets = inputs;
        }
    }

    Ok(lifted)
}

fn clone_bundle(bundle: &TheoryBundle) -> TheoryBundle {
    TheoryBundle {
        object_theory: bundle.object_theory.clone(),
        arrow_theory: bundle.arrow_theory.clone(),
        definitions: clone_definitions(&bundle.definitions),
    }
}

fn clone_definitions(
    definitions: &HashMap<Operation, Declaration>,
) -> HashMap<Operation, Declaration> {
    definitions
        .iter()
        .map(|(name, declaration)| {
            (
                name.clone(),
                Declaration {
                    theory: declaration.theory.clone(),
                    name: declaration.name.clone(),
                    source_map: declaration.source_map.clone(),
                    target_map: declaration.target_map.clone(),
                    definition: declaration.definition.clone(),
                },
            )
        })
        .collect()
}
