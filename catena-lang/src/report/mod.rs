mod elaboration;
mod gpu;
mod svg;

use std::{fs, io, path::Path};

use hexpr::Operation;
use metacat::{
    theory::{RawTheorySet, TheoryId, TheorySet},
    tree::Tree,
};
use std::collections::BTreeMap;

use crate::check::{AnnotatedTerm, PartialDefinitionTypes};
use crate::closure::Conversion;
use crate::codegen::GpuModuleMap;
use crate::pass::record_boundary_sizes::OperationWithBoundarySizes;

/// Generic storage for per-theory, per-definition graph results produced by compiler passes.
pub type TheoryTermMap<A = Operation> = BTreeMap<TheoryId, BTreeMap<Operation, AnnotatedTerm<A>>>;
#[derive(Debug)]
pub struct CompileReport {
    pub raw_theories: RawTheorySet,
    pub elaborated: Option<RawTheorySet>,
    pub theory_set: Option<TheorySet>,
    pub definition_types: Option<BTreeMap<TheoryId, BTreeMap<Operation, Vec<Tree<(), Operation>>>>>,
    pub partial_definition_types: Option<PartialDefinitionTypes>,
    pub closure_conversion: Option<Conversion>,
    pub boundary_sizes: Option<TheoryTermMap<OperationWithBoundarySizes<Operation>>>,
    pub unpacked_products: Option<TheoryTermMap<OperationWithBoundarySizes<Operation>>>,
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
            closure_conversion: None,
            boundary_sizes: None,
            unpacked_products: None,
            gpu_modules: None,
        }
    }
}

impl CompileReport {
    pub fn dump_graphs_to_dir(&self, dir: impl AsRef<Path>) -> io::Result<()> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;
        fs::write(
            dir.join("raw_theories.hex"),
            self.raw_theories.to_hexpr_text(),
        )?;
        elaboration::dump_elaboration(self, dir)?;
        svg::dump_svgs(self, &dir.join("svgs"))?;
        Ok(())
    }

    pub fn dump_to_dir(&self, dir: impl AsRef<Path>) -> io::Result<()> {
        let dir = dir.as_ref();
        self.dump_graphs_to_dir(dir)?;
        gpu::dump_gpu(self, &dir.join("gpu"))?;
        Ok(())
    }
}
