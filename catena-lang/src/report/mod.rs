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
        self.dump_elaboration(dir)?;
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

    fn dump_elaboration(&self, dir: &Path) -> io::Result<()> {
        let elaboration_dir = dir.join("elaboration");
        fs::create_dir_all(&elaboration_dir)?;
        fs::write(
            elaboration_dir.join("input.hex"),
            self.raw_theories.to_hexpr_text(),
        )?;
        if let Some(elaborated) = &self.elaborated {
            fs::write(
                elaboration_dir.join("output.hex"),
                elaborated.to_hexpr_text(),
            )?;
            fs::write(
                elaboration_dir.join("generated.hex"),
                elaboration_delta(&self.raw_theories, elaborated)?.to_hexpr_text(),
            )?;
        }
        Ok(())
    }
}

fn elaboration_delta(raw: &RawTheorySet, elaborated: &RawTheorySet) -> io::Result<RawTheorySet> {
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
