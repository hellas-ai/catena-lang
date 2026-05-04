use hexpr::Operation;
use metacat::{
    check::check,
    theory::{Theory, TheorySet},
};
use open_hypergraphs::lax::OpenHypergraph;
use thiserror::Error;

use crate::compile::{
    config::{CompileConfig, TheoryExtension},
    lift::{LiftError, lift_with_tensor},
};

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
    pub source: OpenHypergraph<(), Operation>,
    pub target: OpenHypergraph<(), Operation>,
}

#[derive(Error, Debug)]
pub enum CheckError {
    #[error("unknown theory `{0}`")]
    UnknownTheory(String),
    #[error("theory `{0}` is not a user theory")]
    NotUserTheory(String),
    #[error("definition {definition} failed typecheck: {error:?}")]
    Typecheck {
        definition: String,
        error: metacat::check::Error<Operation>,
    },
    #[error("{error}")]
    Lift { error: LiftError },
}

impl From<LiftError> for CheckError {
    fn from(error: LiftError) -> Self {
        Self::Lift { error }
    }
}

pub fn check_compile_set(set: &TheorySet) -> Result<CompileCheckReport, CheckError> {
    let config = CompileConfig::data_control();
    let syntax = theory(set, config.syntax)?;
    let data = theory(set, "data")?;
    let data_to_control_extension = extension(&config, "control", "data")?;
    let control_to_data_extension = extension(&config, "data", "control")?;

    let data_report = check_theory(data)?;
    let control_with_data = lift_extension(set, &config, syntax, data_to_control_extension)?;
    let data_with_control = lift_extension(set, &config, syntax, control_to_data_extension)?;
    let control_with_data_report = check_theory(&control_with_data)?;
    let data_with_control_report = check_theory(&data_with_control)?;
    let data_to_control = lifted_arrow_types(&control_with_data, data_to_control_extension.prefix);
    let control_to_data = lifted_arrow_types(&data_with_control, control_to_data_extension.prefix);

    Ok(CompileCheckReport {
        data: data_report,
        control_with_data: control_with_data_report,
        data_with_control: data_with_control_report,
        data_to_control,
        control_to_data,
    })
}

fn extension<'a>(
    config: &'a CompileConfig,
    target: &str,
    prefix: &str,
) -> Result<&'a TheoryExtension, CheckError> {
    config
        .extension_for_target_and_prefix(target, prefix)
        .ok_or_else(|| CheckError::UnknownTheory(format!("{target}/{prefix}")))
}

fn lift_extension(
    set: &TheorySet,
    config: &CompileConfig,
    syntax: &Theory,
    extension: &TheoryExtension,
) -> Result<Theory, CheckError> {
    let source = theory(set, extension.source)?;
    let target = theory(set, extension.target)?;
    let excluded_prefixes = config.lifted_prefixes();
    Ok(lift_with_tensor(
        source,
        target,
        syntax,
        extension.prefix,
        extension.tensor,
        extension.unit,
        &excluded_prefixes,
    )?)
}

pub fn check_theory(theory: &Theory) -> Result<CheckReport, CheckError> {
    let Theory::Theory { arrows, .. } = theory else {
        return Err(CheckError::NotUserTheory("nat".to_string()));
    };

    let mut definitions_checked = 0;
    for (name, arrow) in arrows {
        let Some(mut term) = arrow.definition.clone() else {
            continue;
        };
        check(
            theory,
            arrow.type_maps.0.clone(),
            arrow.type_maps.1.clone(),
            &mut term,
        )
        .map_err(|error| CheckError::Typecheck {
            definition: name.to_string(),
            error,
        })?;
        definitions_checked += 1;
    }

    Ok(CheckReport {
        definitions_checked,
    })
}

fn lifted_arrow_types(theory: &Theory, prefix: &str) -> Vec<ArrowType> {
    let Theory::Theory { arrows, .. } = theory else {
        return Vec::new();
    };
    let mut operations = arrows
        .iter()
        .filter(|(op, _)| op.to_string().starts_with(&format!("{prefix}.")))
        .map(|(op, arrow)| ArrowType {
            name: op.to_string(),
            source: arrow.type_maps.0.clone(),
            target: arrow.type_maps.1.clone(),
        })
        .collect::<Vec<_>>();
    operations.sort_by(|left, right| left.name.cmp(&right.name));
    operations
}

pub fn theory<'a>(set: &'a TheorySet, name: &str) -> Result<&'a Theory, CheckError> {
    let id = metacat::theory::TheoryId(
        name.parse()
            .map_err(|_| CheckError::UnknownTheory(name.to_string()))?,
    );
    set.theories
        .get(&id)
        .ok_or_else(|| CheckError::UnknownTheory(name.to_string()))
}
