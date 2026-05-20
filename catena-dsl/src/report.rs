mod c;
mod svg;

use std::{fs, io, path::Path};

use catena::structured::StructuredProgram;
use hexpr::Operation;
use metacat::{
    theory::{RawTheorySet, TheoryId, TheorySet},
    tree::Tree,
};
use open_hypergraphs::lax::OpenHypergraph;
use std::collections::BTreeMap;

/// A definition graph whose nodes are annotated with their computed object types.
pub type AnnotatedTerm = OpenHypergraph<Tree<(), Operation>, Operation>;
/// Generic storage for per-theory, per-definition graph results produced by compiler passes.
pub type TheoryTermMap = BTreeMap<TheoryId, BTreeMap<Operation, AnnotatedTerm>>;
/// Generic storage for per-theory, per-definition structured codegen results.
pub type StructuredProgramMap = BTreeMap<TheoryId, BTreeMap<Operation, StructuredProgram>>;

pub struct CompileReport {
    pub raw_theories: RawTheorySet,
    pub elaborated: RawTheorySet,
    pub theory_set: TheorySet,
    pub definition_types: BTreeMap<TheoryId, BTreeMap<Operation, Vec<Tree<(), Operation>>>>,
    pub forgotten_closures: TheoryTermMap,
    pub structured_programs: StructuredProgramMap,
}

impl CompileReport {
    pub fn dump_to_dir(&self, dir: impl AsRef<Path>) -> io::Result<()> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;
        fs::write(
            dir.join("raw_theories.hex"),
            self.raw_theories.to_hexpr_text(),
        )?;
        fs::write(dir.join("elaborated.hex"), self.elaborated.to_hexpr_text())?;
        svg::dump_svgs(self, &dir.join("svgs"))?;
        c::dump_c(self, &dir.join("c"))?;
        Ok(())
    }
}
