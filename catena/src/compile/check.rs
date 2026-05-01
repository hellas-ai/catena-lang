use hexpr::try_interpret;
use metacat::{check::check, syntax::TheoryBundle, theory::OperationKey};
use open_hypergraphs::lax::OpenHypergraph;
use thiserror::Error;

use crate::compile::lift::{LiftError, lift_control_to_data, lift_data_to_control};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckReport {
    pub definitions_checked: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompileCheckReport {
    pub data: CheckReport,
    pub control_with_data: CheckReport,
    pub data_with_control: CheckReport,
    pub data_to_control: Vec<ArrowType>,
    pub control_to_data: Vec<ArrowType>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ArrowType {
    pub name: String,
    pub source: OpenHypergraph<(), OperationKey>,
    pub target: OpenHypergraph<(), OperationKey>,
}

#[derive(Error, Debug)]
pub enum CheckError {
    #[error("invalid hexpr: {0}")]
    InvalidHexpr(#[from] hexpr::interpret::Error<metacat::theory::Error>),
    #[error("definition {definition} failed typecheck: {error:?}")]
    Typecheck {
        definition: String,
        error: metacat::check::Error<OperationKey>,
    },
    #[error("{error}")]
    Lift { error: LiftError },
}

impl From<LiftError> for CheckError {
    fn from(error: LiftError) -> Self {
        Self::Lift { error }
    }
}

pub fn check_compile_bundle(
    data: &TheoryBundle,
    control: &TheoryBundle,
) -> Result<CompileCheckReport, CheckError> {
    let data_report = check_bundle(data)?;
    let control_with_data = lift_data_to_control(data, control)?;
    let data_with_control = lift_control_to_data(control, data)?;
    let control_with_data_report = check_bundle(&control_with_data)?;
    let data_with_control_report = check_bundle(&data_with_control)?;
    let data_to_control = lifted_arrow_types(&control_with_data, "data");
    let control_to_data = lifted_arrow_types(&data_with_control, "control");

    Ok(CompileCheckReport {
        data: data_report,
        control_with_data: control_with_data_report,
        data_with_control: data_with_control_report,
        data_to_control,
        control_to_data,
    })
}

fn lifted_arrow_types(bundle: &TheoryBundle, prefix: &str) -> Vec<ArrowType> {
    let mut operations: Vec<_> = bundle
        .arrow_theory
        .operations()
        .filter(|op| op.to_string().starts_with(&format!("{prefix}.")))
        .map(|op| {
            let (source, target) = bundle.arrow_theory.type_maps(op);
            ArrowType {
                name: op.to_string(),
                source: source.clone(),
                target: target.clone(),
            }
        })
        .collect();
    operations.sort_by(|left, right| left.name.cmp(&right.name));
    operations
}

pub fn check_bundle(bundle: &TheoryBundle) -> Result<CheckReport, CheckError> {
    let mut definitions: Vec<_> = bundle.definitions.iter().collect();
    definitions.sort_by_key(|(name, _)| name.to_string());

    let mut definitions_checked = 0;
    for (name, declaration) in definitions {
        let hexpr = declaration
            .definition
            .as_ref()
            .expect("definition entries always have a body");
        let mut term = forget_labels(try_interpret(&bundle.arrow_theory, hexpr)?);
        let source = forget_labels(try_interpret(
            &bundle.object_theory,
            &declaration.source_map,
        )?);
        let target = forget_labels(try_interpret(
            &bundle.object_theory,
            &declaration.target_map,
        )?);

        check(&bundle.arrow_theory, source, target, &mut term).map_err(|error| {
            CheckError::Typecheck {
                definition: name.to_string(),
                error,
            }
        })?;
        definitions_checked += 1;
    }

    Ok(CheckReport {
        definitions_checked,
    })
}

fn forget_labels<O, A>(f: OpenHypergraph<O, A>) -> OpenHypergraph<(), A> {
    f.map_nodes(|_| ())
}
