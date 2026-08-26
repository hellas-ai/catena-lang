use std::collections::BTreeMap;

use hexpr::Operation;
use metacat::{
    check::check as metacat_check,
    theory::{Theory, TheoryId, TheorySet},
    tree::Tree,
};
use open_hypergraphs::lax::OpenHypergraph;
use thiserror::Error;

pub type AnnotatedTerm<A = Operation> = OpenHypergraph<Tree<(), Operation>, A>;
pub type DefinitionTypes = BTreeMap<TheoryId, BTreeMap<Operation, Vec<Tree<(), Operation>>>>;

#[derive(Debug, Error)]
pub enum CheckError {
    #[error("definition check failed in theory `{theory}`, definition `{definition}`: {error:?}")]
    Definition {
        theory: String,
        definition: String,
        error: metacat::check::Error<Operation>,
    },
    #[error("missing checked node types for `{theory}.{definition}`")]
    MissingTypes { theory: String, definition: String },
    #[error("checked node count does not match `{theory}.{definition}`")]
    NodeCount { theory: String, definition: String },
}

pub fn check(theory_set: &TheorySet) -> Result<DefinitionTypes, CheckError> {
    let mut output = BTreeMap::new();
    for (id, theory) in &theory_set.theories {
        let Theory::Theory { arrows, .. } = theory else {
            continue;
        };
        let mut definitions = BTreeMap::new();
        for (name, arrow) in arrows {
            let Some(mut body) = arrow.definition.clone() else {
                continue;
            };
            let types = metacat_check(
                &theory_set.theories[id],
                arrow.type_maps.0.clone(),
                arrow.type_maps.1.clone(),
                &mut body,
            )
            .map_err(|error| CheckError::Definition {
                theory: id.to_string(),
                definition: name.to_string(),
                error,
            })?;
            definitions.insert(name.clone(), types);
        }
        if !definitions.is_empty() {
            output.insert(id.clone(), definitions);
        }
    }
    Ok(output)
}

pub(crate) fn typed_definitions(
    theory_set: &TheorySet,
    types: &DefinitionTypes,
) -> Result<crate::report::TheoryTermMap, CheckError> {
    let mut output = BTreeMap::new();
    for (id, theory) in &theory_set.theories {
        let Theory::Theory { arrows, .. } = theory else {
            continue;
        };
        let mut definitions = BTreeMap::new();
        for (name, arrow) in arrows {
            let Some(mut body) = arrow.definition.clone() else {
                continue;
            };
            body.quotient().ok();
            let labels = types
                .get(id)
                .and_then(|definitions| definitions.get(name))
                .cloned()
                .ok_or_else(|| CheckError::MissingTypes {
                    theory: id.to_string(),
                    definition: name.to_string(),
                })?;
            let body = body
                .with_nodes(|_| labels)
                .ok_or_else(|| CheckError::NodeCount {
                    theory: id.to_string(),
                    definition: name.to_string(),
                })?;
            definitions.insert(name.clone(), body);
        }
        if !definitions.is_empty() {
            output.insert(id.clone(), definitions);
        }
    }
    Ok(output)
}
