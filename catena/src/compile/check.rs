use hexpr::Operation;
use metacat::{
    check::check,
    theory::{Theory, TheoryId, TheorySet},
};
use open_hypergraphs::lax::OpenHypergraph;
use thiserror::Error;

use crate::compile::{config::CompileConfig, lift::LiftError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckReport {
    pub definitions_checked: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompileCheckReport {
    pub theories: Vec<TheoryCheckReport>,
    pub extensions: Vec<ExtensionCheckReport>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TheoryCheckReport {
    pub name: String,
    pub report: CheckReport,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExtensionCheckReport {
    pub target: String,
    pub source: String,
    pub prefix: String,
    pub arrows: Vec<ArrowType>,
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

/// Check a compile theory set that has already been extended according to
/// `config`, for example by `load_extended_theory_set_from_text`.
pub fn check_compile_theories(
    set: &TheorySet,
    config: &CompileConfig,
) -> Result<CompileCheckReport, CheckError> {
    let mut theories = Vec::new();
    for (id, loaded_theory) in &set.theories {
        let name = id.to_string();
        if name == "nat" || name == config.syntax {
            continue;
        }
        theories.push(TheoryCheckReport {
            name,
            report: check_theory(loaded_theory)?,
        });
    }

    let extensions = config
        .extensions
        .iter()
        .map(|extension| {
            let target = theory(set, extension.target)?;
            Ok(ExtensionCheckReport {
                target: extension.target.to_string(),
                source: extension.source.to_string(),
                prefix: extension.prefix.to_string(),
                arrows: lifted_arrow_types(target, extension.prefix),
            })
        })
        .collect::<Result<Vec<_>, CheckError>>()?;

    Ok(CompileCheckReport {
        theories,
        extensions,
    })
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
    let id = TheoryId(
        name.parse()
            .map_err(|_| CheckError::UnknownTheory(name.to_string()))?,
    );
    set.theories
        .get(&id)
        .ok_or_else(|| CheckError::UnknownTheory(name.to_string()))
}
