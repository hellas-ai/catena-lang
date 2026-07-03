//! Inline a specified set of definitions inside each theory.

use std::collections::{BTreeMap, BTreeSet};

use hexpr::Operation;
use metacat::theory::{Theory, TheoryId, TheorySet};
use open_hypergraphs::lax::{
    OpenHypergraph,
    functor::{Functor, try_define_map_arrow},
};
use thiserror::Error;

use crate::prefixes::NAME_PREFIX;

pub type Term = OpenHypergraph<(), Operation>;

#[derive(Debug, Error)]
pub enum InlineDefinitionsError {
    #[error("missing theory `{0}`")]
    MissingTheory(String),
    #[error("theory `{0}` is not a user theory")]
    NotUserTheory(String),
    #[error("missing definition `{definition}` in theory `{theory}`")]
    MissingDefinition { theory: String, definition: String },
    #[error("cyclic inline dependency in theory `{theory}` at definition `{definition}`")]
    Cycle { theory: String, definition: String },
    #[error("failed to inline definition `{definition}` in theory `{theory}`")]
    Inline { theory: String, definition: String },
}

/// Return a copy of `theory_set` where every selected definition is inlined into
/// definitions that use it, and selected definitions are removed from their theories.
///
/// Dependencies are resolved bottom-up, so if an inlined definition uses another
/// selected definition, its collected inline body already contains that expansion.
pub fn run(
    theory_set: &TheorySet,
    definitions_to_inline: &BTreeMap<TheoryId, BTreeSet<Operation>>,
) -> Result<TheorySet, InlineDefinitionsError> {
    let mut output = theory_set.clone();

    for (theory_id, selected) in definitions_to_inline {
        if selected.is_empty() {
            continue;
        }

        let theory = theory_set
            .theories
            .get(theory_id)
            .ok_or_else(|| InlineDefinitionsError::MissingTheory(theory_id.to_string()))?;
        let Theory::Theory { arrows, .. } = theory else {
            return Err(InlineDefinitionsError::NotUserTheory(theory_id.to_string()));
        };

        let inline_bodies = collect_inline_bodies(theory_id, arrows, selected)?;

        let Some(Theory::Theory { arrows, .. }) = output.theories.get_mut(theory_id) else {
            unreachable!("validated user theory should exist in cloned output");
        };

        for (definition_name, arrow) in arrows.iter_mut() {
            if selected.contains(definition_name) {
                continue;
            }

            let Some(definition) = arrow.definition.clone() else {
                continue;
            };

            arrow.definition = Some(inline_term(
                theory_id,
                definition_name,
                definition,
                &inline_bodies,
            )?);
        }

        for definition_name in selected {
            arrows.remove(definition_name);
            arrows.remove(&name_operation(definition_name));
        }
    }

    Ok(output)
}

fn collect_inline_bodies(
    theory_id: &TheoryId,
    arrows: &BTreeMap<Operation, metacat::theory::TheoryArrow>,
    selected: &BTreeSet<Operation>,
) -> Result<BTreeMap<Operation, Term>, InlineDefinitionsError> {
    let mut states = BTreeMap::new();
    let mut inline_bodies = BTreeMap::new();

    for definition_name in selected {
        collect_inline_body(
            theory_id,
            arrows,
            selected,
            definition_name,
            &mut states,
            &mut inline_bodies,
        )?;
    }

    Ok(inline_bodies)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum VisitState {
    Visiting,
    Done,
}

fn collect_inline_body(
    theory_id: &TheoryId,
    arrows: &BTreeMap<Operation, metacat::theory::TheoryArrow>,
    selected: &BTreeSet<Operation>,
    definition_name: &Operation,
    states: &mut BTreeMap<Operation, VisitState>,
    inline_bodies: &mut BTreeMap<Operation, Term>,
) -> Result<(), InlineDefinitionsError> {
    match states.get(definition_name) {
        Some(VisitState::Done) => return Ok(()),
        Some(VisitState::Visiting) => {
            return Err(InlineDefinitionsError::Cycle {
                theory: theory_id.to_string(),
                definition: definition_name.to_string(),
            });
        }
        None => {}
    }

    let arrow = arrows
        .get(definition_name)
        .and_then(|arrow| arrow.definition.clone())
        .ok_or_else(|| InlineDefinitionsError::MissingDefinition {
            theory: theory_id.to_string(),
            definition: definition_name.to_string(),
        })?;

    states.insert(definition_name.clone(), VisitState::Visiting);
    for dependency in selected_dependencies(&arrow, selected) {
        collect_inline_body(
            theory_id,
            arrows,
            selected,
            &dependency,
            states,
            inline_bodies,
        )?;
    }

    let inlined = inline_term(theory_id, definition_name, arrow, inline_bodies)?;
    inline_bodies.insert(definition_name.clone(), inlined);
    states.insert(definition_name.clone(), VisitState::Done);
    Ok(())
}

fn selected_dependencies(term: &Term, selected: &BTreeSet<Operation>) -> BTreeSet<Operation> {
    term.hypergraph
        .edges
        .iter()
        .filter(|operation| selected.contains(*operation))
        .cloned()
        .collect()
}

fn name_operation(definition_name: &Operation) -> Operation {
    format!("{NAME_PREFIX}{definition_name}")
        .parse()
        .expect("generated name operation should parse")
}

fn inline_term(
    theory_id: &TheoryId,
    definition_name: &Operation,
    mut term: Term,
    definitions: &BTreeMap<Operation, Term>,
) -> Result<Term, InlineDefinitionsError> {
    term.quotient().ok();
    let mut output =
        try_define_map_arrow(&InlineDefinitions { definitions }, &term).ok_or_else(|| {
            InlineDefinitionsError::Inline {
                theory: theory_id.to_string(),
                definition: definition_name.to_string(),
            }
        })?;
    output.quotient().ok();
    Ok(output)
}

struct InlineDefinitions<'a> {
    definitions: &'a BTreeMap<Operation, Term>,
}

impl Functor<(), Operation, (), Operation> for InlineDefinitions<'_> {
    fn map_object(&self, _o: &()) -> impl ExactSizeIterator<Item = ()> {
        std::iter::once(())
    }

    fn map_operation(
        &self,
        operation: &Operation,
        source: &[()],
        target: &[()],
    ) -> OpenHypergraph<(), Operation> {
        self.definitions.get(operation).cloned().unwrap_or_else(|| {
            OpenHypergraph::singleton(operation.clone(), source.to_vec(), target.to_vec())
        })
    }

    fn map_arrow(&self, term: &Term) -> Term {
        try_define_map_arrow(self, term).expect("quotiented term should be inlineable")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{elaborate::elaborate, stdlib};
    use metacat::theory::RawTheorySet;

    #[test]
    fn inlines_and_removes_selected_definitions() {
        let theory_set = theory_set(
            r#"
            (def program mk-closure : (f32 val) -> ({1 (f32 val)} =>) = defer)
            (def program use-closure : (f32 val) -> (f32 val) = (mk-closure run))
            "#,
        );
        let selected = selected_program_definitions(["mk-closure"]);

        let output = run(&theory_set, &selected).expect("inline pass should succeed");
        let program = output
            .theories
            .get(&TheoryId("program".parse().unwrap()))
            .expect("program theory should exist");
        let Theory::Theory { arrows, .. } = program else {
            panic!("program should be a user theory");
        };

        assert!(!arrows.contains_key(&"mk-closure".parse().unwrap()));
        assert!(!arrows.contains_key(&"name.mk-closure".parse().unwrap()));
        let use_closure = arrows
            .get(&"use-closure".parse().unwrap())
            .and_then(|arrow| arrow.definition.as_ref())
            .expect("use-closure should remain defined");
        assert!(
            !use_closure
                .hypergraph
                .edges
                .iter()
                .any(|operation| operation.as_str() == "mk-closure")
        );
        assert!(
            use_closure
                .hypergraph
                .edges
                .iter()
                .any(|operation| operation.as_str() == "defer")
        );
    }

    #[test]
    fn selected_definitions_are_inlined_bottom_up() {
        let theory_set = theory_set(
            r#"
            (def program mk-closure : (f32 val) -> ({1 (f32 val)} =>) = defer)
            (def program mk-closure2 : (f32 val) -> ({1 (f32 val)} =>) = mk-closure)
            (def program use-closure : (f32 val) -> (f32 val) = (mk-closure2 run))
            "#,
        );
        let selected = selected_program_definitions(["mk-closure", "mk-closure2"]);

        let output = run(&theory_set, &selected).expect("inline pass should succeed");
        let program = output
            .theories
            .get(&TheoryId("program".parse().unwrap()))
            .expect("program theory should exist");
        let Theory::Theory { arrows, .. } = program else {
            panic!("program should be a user theory");
        };
        let use_closure = arrows
            .get(&"use-closure".parse().unwrap())
            .and_then(|arrow| arrow.definition.as_ref())
            .expect("use-closure should remain defined");

        assert!(
            use_closure
                .hypergraph
                .edges
                .iter()
                .all(|operation| operation.as_str() != "mk-closure"
                    && operation.as_str() != "mk-closure2")
        );
        assert!(
            use_closure
                .hypergraph
                .edges
                .iter()
                .any(|operation| operation.as_str() == "defer")
        );
    }

    #[test]
    fn errors_when_selected_theory_is_missing() {
        let theory_set = theory_set("");
        let selected = selected_definitions("missing", ["mk-closure"]);

        let error = run(&theory_set, &selected).expect_err("missing theory should error");

        assert!(matches!(
            error,
            InlineDefinitionsError::MissingTheory(theory) if theory == "missing"
        ));
    }

    #[test]
    fn errors_when_selected_theory_is_not_user_theory() {
        let theory_set = theory_set("");
        let selected = selected_definitions("nat", ["mk-closure"]);

        let error = run(&theory_set, &selected).expect_err("nat theory should error");

        assert!(matches!(
            error,
            InlineDefinitionsError::NotUserTheory(theory) if theory == "nat"
        ));
    }

    fn theory_set(source: &'static str) -> TheorySet {
        let raw = RawTheorySet::from_texts(stdlib::sources().chain([source]))
            .expect("test theories should parse");
        let elaborated = elaborate(raw).expect("test theories should elaborate");
        TheorySet::from_raw(elaborated).expect("test theories should interpret")
    }

    fn selected_program_definitions(
        definitions: impl IntoIterator<Item = &'static str>,
    ) -> BTreeMap<TheoryId, BTreeSet<Operation>> {
        selected_definitions("program", definitions)
    }

    fn selected_definitions(
        theory: &'static str,
        definitions: impl IntoIterator<Item = &'static str>,
    ) -> BTreeMap<TheoryId, BTreeSet<Operation>> {
        BTreeMap::from([(
            TheoryId(theory.parse().unwrap()),
            definitions
                .into_iter()
                .map(|definition| definition.parse().unwrap())
                .collect(),
        )])
    }
}
