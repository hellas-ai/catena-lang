use std::collections::BTreeMap;

use hexpr::Operation;
use metacat::{
    check::{
        Error as MetacatCheckError, PartialResult as MetacatPartialResult, check as metacat_check,
    },
    theory::{Theory, TheoryId, TheorySet},
    tree::Tree,
};
use thiserror::Error;

pub type DefinitionTypes = BTreeMap<TheoryId, BTreeMap<Operation, Vec<Tree<(), Operation>>>>;
pub type PartialDefinitionTypes =
    BTreeMap<TheoryId, BTreeMap<Operation, Vec<Option<Tree<(), Operation>>>>>;

#[derive(Debug, Error)]
pub enum CheckError {
    #[error("definition check failed in theory `{theory}`, definition `{definition}`: {error:?}")]
    Definition {
        theory: String,
        definition: String,
        error: metacat::check::Error<Operation>,
    },
}

pub fn partial_definition_types(error: &CheckError) -> Option<PartialDefinitionTypes> {
    let CheckError::Definition {
        theory,
        definition,
        error: MetacatCheckError::PartialResult(MetacatPartialResult { partial_result, .. }),
    } = error
    else {
        return None;
    };

    let mut theory_defs = BTreeMap::new();
    theory_defs.insert(definition.parse().ok()?, partial_result.clone());

    let mut out = BTreeMap::new();
    out.insert(TheoryId(theory.parse().ok()?), theory_defs);
    Some(out)
}

pub fn check(theory_set: &TheorySet) -> Result<DefinitionTypes, CheckError> {
    let mut definition_types = BTreeMap::new();

    for (id, theory) in &theory_set.theories {
        let Theory::Theory { arrows, .. } = theory else {
            continue;
        };

        let mut theory_definition_types = BTreeMap::new();
        for (name, arrow) in arrows {
            let Some(mut body) = arrow.definition.clone() else {
                continue;
            };

            let types = metacat_check(
                theory,
                arrow.type_maps.0.clone(),
                arrow.type_maps.1.clone(),
                &mut body,
            )
            .map_err(|error| CheckError::Definition {
                theory: id.to_string(),
                definition: name.to_string(),
                error,
            })?;

            theory_definition_types.insert(name.clone(), types);
        }

        if !theory_definition_types.is_empty() {
            definition_types.insert(id.clone(), theory_definition_types);
        }
    }

    Ok(definition_types)
}
