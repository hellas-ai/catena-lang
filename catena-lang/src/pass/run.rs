use std::collections::BTreeMap;

use hexpr::Operation;
use metacat::{
    theory::{Theory, TheoryId, TheorySet},
    tree::Tree,
};
use thiserror::Error;

use crate::{
    check::DefinitionTypes,
    pass::passes,
    pass::record_object_sizes::OperationWithSizes,
    report::{AnnotatedTerm, TheoryTermMap},
};

#[derive(Debug, Error)]
pub enum PassRunError {
    #[error("missing definition `{definition}` in theory `{theory}`")]
    MissingDefinition { theory: String, definition: String },
    #[error("missing checked node types for definition `{definition}` in theory `{theory}`")]
    MissingDefinitionTypes { theory: String, definition: String },
    #[error(
        "typechecked node label count mismatch for definition `{definition}` in theory `{theory}`"
    )]
    NodeLabelCountMismatch { theory: String, definition: String },
}

pub fn run(
    theory_set: &TheorySet,
    definition_types: &DefinitionTypes,
) -> Result<TheoryTermMap<OperationWithSizes<Operation>>, PassRunError> {
    let mut output = BTreeMap::new();

    for (theory_id, theory) in &theory_set.theories {
        let Theory::Theory { arrows, .. } = theory else {
            continue;
        };

        let mut transformed = BTreeMap::new();
        let theory_definition_types = definition_types.get(theory_id);
        for (definition_name, arrow) in arrows {
            let Some(_) = &arrow.definition else {
                continue;
            };

            let typed =
                typed_definition(theory_id, definition_name, theory, theory_definition_types)?;
            transformed.insert(definition_name.clone(), passes::apply(&typed));
        }

        if !transformed.is_empty() {
            output.insert(theory_id.clone(), transformed);
        }
    }

    Ok(output)
}

fn typed_definition(
    theory_id: &TheoryId,
    definition_name: &Operation,
    theory: &Theory,
    theory_definition_types: Option<&BTreeMap<Operation, Vec<Tree<(), Operation>>>>,
) -> Result<AnnotatedTerm, PassRunError> {
    let Theory::Theory { arrows, .. } = theory else {
        unreachable!("typed_definition only called on user theories");
    };
    let arrow = arrows
        .get(definition_name)
        .expect("definition should exist in current theory");
    let body = arrow
        .definition
        .clone()
        .ok_or_else(|| PassRunError::MissingDefinition {
            theory: theory_id.to_string(),
            definition: definition_name.to_string(),
        })?;
    let mut body = body;
    body.quotient().ok();
    let labels = theory_definition_types
        .and_then(|types| types.get(definition_name))
        .cloned()
        .ok_or_else(|| PassRunError::MissingDefinitionTypes {
            theory: theory_id.to_string(),
            definition: definition_name.to_string(),
        })?;
    body.with_nodes(|_| labels)
        .ok_or_else(|| PassRunError::NodeLabelCountMismatch {
            theory: theory_id.to_string(),
            definition: definition_name.to_string(),
        })
}
