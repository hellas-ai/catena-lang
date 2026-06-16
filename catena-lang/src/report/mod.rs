mod gpu;
mod svg;

use std::{fs, io, path::Path};

use hexpr::Operation;
use metacat::{
    theory::{RawTheorySet, TheoryId, TheorySet, ast::RawTheory},
    tree::Tree,
};
use open_hypergraphs::lax::OpenHypergraph;
use std::collections::BTreeMap;

use crate::check::PartialDefinitionTypes;
use crate::codegen::GpuModuleMap;

/// A definition graph whose nodes are annotated with their computed object types.
pub type AnnotatedTerm = OpenHypergraph<Tree<(), Operation>, Operation>;
/// Generic storage for per-theory, per-definition graph results produced by compiler passes.
pub type TheoryTermMap = BTreeMap<TheoryId, BTreeMap<Operation, AnnotatedTerm>>;
#[derive(Debug)]
pub struct CompileReport {
    pub raw_theories: RawTheorySet,
    pub elaborated: Option<RawTheorySet>,
    pub theory_set: Option<TheorySet>,
    pub definition_types: Option<BTreeMap<TheoryId, BTreeMap<Operation, Vec<Tree<(), Operation>>>>>,
    pub partial_definition_types: Option<PartialDefinitionTypes>,
    pub forgotten_closures: Option<TheoryTermMap>,
    pub gpu_modules: Option<GpuModuleMap>,
}

impl CompileReport {
    pub fn new(raw_theories: RawTheorySet) -> Self {
        Self {
            raw_theories,
            elaborated: None,
            theory_set: None,
            definition_types: None,
            partial_definition_types: None,
            forgotten_closures: None,
            gpu_modules: None,
        }
    }
}

impl CompileReport {
    pub fn dump_to_dir(&self, dir: impl AsRef<Path>) -> io::Result<()> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;
        self.dump_generated_elaboration(dir)?;
        fs::write(
            dir.join("raw_theories.hex"),
            self.raw_theories.to_hexpr_text(),
        )?;
        if let Some(elaborated) = &self.elaborated {
            fs::write(dir.join("elaborated.hex"), elaborated.to_hexpr_text())?;
        }
        svg::dump_svgs(self, &dir.join("svgs"))?;
        gpu::dump_gpu(self, &dir.join("gpu"))?;
        Ok(())
    }

    fn dump_generated_elaboration(&self, dir: &Path) -> io::Result<()> {
        let Some(elaborated) = &self.elaborated else {
            return Ok(());
        };

        let elaboration_dir = dir.join("elaboration");
        fs::create_dir_all(&elaboration_dir)?;
        fs::write(
            elaboration_dir.join("generated.hex"),
            generated_elaboration(&self.raw_theories, elaborated)?.to_hexpr_text(),
        )?;
        Ok(())
    }
}

fn generated_elaboration(
    raw: &RawTheorySet,
    elaborated: &RawTheorySet,
) -> io::Result<RawTheorySet> {
    let baseline = raw
        .clone()
        .with_extensions()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let mut theories = BTreeMap::new();

    for (theory_name, elaborated_theory) in &elaborated.theories {
        let baseline_theory = baseline.theories.get(theory_name);
        let mut arrows = BTreeMap::new();

        for (arrow_name, arrow) in &elaborated_theory.arrows {
            let existed_before =
                baseline_theory.is_some_and(|theory| theory.arrows.contains_key(arrow_name));
            if !existed_before {
                arrows.insert(arrow_name.clone(), arrow.clone());
            }
        }

        if !arrows.is_empty() {
            theories.insert(
                theory_name.clone(),
                RawTheory {
                    name: elaborated_theory.name.clone(),
                    syntax_category: elaborated_theory.syntax_category.clone(),
                    arrows,
                },
            );
        }
    }

    Ok(RawTheorySet {
        theories,
        extensions: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use metacat::theory::RawTheorySet;

    use crate::elaborate::elaborate;

    #[test]
    fn generated_elaboration_file_contains_only_elaborated_arrows() {
        let raw = RawTheorySet::from_text(
            r#"
            (theory type nat {
              (arr 1 : 0 -> 1)
              (arr bool : 0 -> 1)
              (arr val : 1 -> 1)
            })

            (theory program type {
              (arr id_bool : (bool val) -> (bool val))
            })
            "#,
        )
        .expect("test theory should parse");

        let elaborated = elaborate(raw.clone()).expect("test theory should elaborate");
        let mut report = super::CompileReport::new(raw);
        report.elaborated = Some(elaborated);

        let dir = tempfile::tempdir().expect("temp dir should be created");
        report.dump_to_dir(dir.path()).expect("report should dump");

        let generated_path = dir.path().join("elaboration").join("generated.hex");
        let generated =
            std::fs::read_to_string(generated_path).expect("generated.hex should exist");

        assert!(generated.contains("name.id_bool"));
        assert!(!generated.contains("(arr id_bool"));

        let generated_raw =
            RawTheorySet::from_text(&generated).expect("generated.hex should parse");
        let program: hexpr::Operation = "program".parse().unwrap();
        let name_id_bool: hexpr::Operation = "name.id_bool".parse().unwrap();
        let id_bool: hexpr::Operation = "id_bool".parse().unwrap();
        let program_theory = generated_raw
            .theories
            .get(&program)
            .expect("program theory should exist");
        assert!(program_theory.arrows.contains_key(&name_id_bool));
        assert!(!program_theory.arrows.contains_key(&id_bool));
    }
}
