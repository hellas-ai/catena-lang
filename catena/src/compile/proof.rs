use std::{collections::BTreeSet, path::PathBuf};

use metacat::{
    check::check as metacat_check,
    theory::{
        RawTheorySet, Theory, TheorySet,
        ast::{MergeRawError, ParseRawError},
    },
};
use thiserror::Error;

use crate::compile::CompileGraph;

mod gpu;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofCertificates {
    checked_definitions: BTreeSet<String>,
}

#[derive(Debug, Error)]
pub enum ProofCertificateError {
    #[error(
        "missing proof certificate: pass one or more --proof <path> files, or pass --no-proof to compile without checking proof certificates"
    )]
    MissingProof,
    #[error("failed to parse proof input: {0}")]
    Parse(ParseRawError),
    #[error("failed to merge proof inputs: {0}")]
    Merge(#[from] MergeRawError),
    #[error("failed to load proof certificate: {0}")]
    Load(#[from] metacat::theory::LoadError),
    #[error("proof check failed in theory `{theory}`, definition `{definition}`: {error:?}")]
    Definition {
        theory: String,
        definition: String,
        error: metacat::check::Error<hexpr::Operation>,
    },
    #[error("no proof of {property} for `{subject}`; expected proof `{expected_proof}`")]
    MissingPropertyProof {
        property: &'static str,
        subject: String,
        expected_proof: String,
    },
}

impl ProofCertificates {
    pub fn from_files(
        program_paths: &[PathBuf],
        proof_paths: &[PathBuf],
    ) -> Result<Self, ProofCertificateError> {
        if proof_paths.is_empty() {
            return Err(ProofCertificateError::MissingProof);
        }

        let theories = load_metacat_proof_theories(program_paths, proof_paths)?;
        let checked_definitions = check_metacat_definitions(&theories)?;
        Ok(Self {
            checked_definitions,
        })
    }

    pub fn verify_graph_properties(
        &self,
        graph: &CompileGraph,
    ) -> Result<ProofEvidence, ProofCertificateError> {
        let checks = ProofPropertyChecks::default();
        checks.verify(self, graph)
    }

    fn contains_definition(&self, name: &str) -> bool {
        self.checked_definitions.contains(name)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProofEvidence {
    requirements: Vec<ProofRequirement>,
}

impl ProofEvidence {
    pub fn requirements(&self) -> &[ProofRequirement] {
        &self.requirements
    }

    pub fn has_definition(&self, definition: &str) -> bool {
        self.requirements
            .iter()
            .any(|requirement| requirement.definition == definition)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofRequirement {
    pub property: &'static str,
    pub subject: String,
    pub definition: String,
}

impl ProofRequirement {
    fn new(property: &'static str, subject: String, definition: String) -> Self {
        Self {
            property,
            subject,
            definition,
        }
    }
}

trait ProofProperty {
    fn requirements(&self, graph: &CompileGraph) -> Vec<ProofRequirement>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ProofPropertyChecks {
    gpu_memory_safety: gpu::MemorySafety,
}

impl Default for ProofPropertyChecks {
    fn default() -> Self {
        Self {
            gpu_memory_safety: gpu::MemorySafety,
        }
    }
}

impl ProofPropertyChecks {
    fn verify(
        &self,
        certificates: &ProofCertificates,
        graph: &CompileGraph,
    ) -> Result<ProofEvidence, ProofCertificateError> {
        let mut evidence = ProofEvidence::default();
        for requirement in self.requirements(graph) {
            if !certificates.contains_definition(&requirement.definition) {
                return Err(ProofCertificateError::MissingPropertyProof {
                    property: requirement.property,
                    subject: requirement.subject,
                    expected_proof: requirement.definition,
                });
            }
            evidence.requirements.push(requirement);
        }
        Ok(evidence)
    }

    fn requirements(&self, graph: &CompileGraph) -> Vec<ProofRequirement> {
        let mut requirements = Vec::new();
        requirements.extend(self.gpu_memory_safety.requirements(graph));
        requirements
    }
}

fn load_metacat_proof_theories(
    program_paths: &[PathBuf],
    proof_paths: &[PathBuf],
) -> Result<TheorySet, ProofCertificateError> {
    let mut raw = RawTheorySet {
        theories: Default::default(),
        extensions: Vec::new(),
    };

    for path in program_paths {
        raw = raw.merge(signature_only(
            RawTheorySet::from_file(path.clone()).map_err(ProofCertificateError::Parse)?,
        ))?;
    }

    for path in proof_paths {
        raw = raw
            .merge(RawTheorySet::from_file(path.clone()).map_err(ProofCertificateError::Parse)?)?;
    }

    Ok(TheorySet::from_raw(raw)?)
}

fn signature_only(mut raw: RawTheorySet) -> RawTheorySet {
    for theory in raw.theories.values_mut() {
        for arrow in theory.arrows.values_mut() {
            arrow.definition = None;
        }
    }
    raw.extensions.clear();
    raw
}

fn check_metacat_definitions(
    theories: &TheorySet,
) -> Result<BTreeSet<String>, ProofCertificateError> {
    let mut checked = BTreeSet::new();

    for (id, theory) in &theories.theories {
        if id.0.as_str() == "syntax" || id.0.as_str() == "nat" {
            continue;
        }

        let Theory::Theory { arrows, .. } = theory else {
            continue;
        };

        for (name, arrow) in arrows {
            let Some(mut body) = arrow.definition.clone() else {
                continue;
            };
            metacat_check(
                theory,
                arrow.type_maps.0.clone(),
                arrow.type_maps.1.clone(),
                &mut body,
            )
            .map_err(|error| ProofCertificateError::Definition {
                theory: id.to_string(),
                definition: name.to_string(),
                error,
            })?;
            checked.insert(name.to_string());
        }
    }

    Ok(checked)
}
