use std::path::PathBuf;

use metacat::theory::{RawTheorySet, TheorySet};
use thiserror::Error;

use crate::{
    check::{CheckError, check as check_elaborated},
    compile::{
        CompileConfig, CompileGraph, CompileGraphError, GraphCompileOptions,
        compile_graph_with_options,
        cuda::{CudaCompileError, CudaOutput, compile_cuda_from_graph},
        graph_render,
    },
    elaborate::{ElaborateError, elaborate},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Emit {
    Cuda,
    CompileGraph,
    Elaborated,
    Checked,
    StructuredIr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    Svg,
    Text,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileRequest {
    pub paths: Vec<PathBuf>,
    pub emit: Emit,
    pub theory: Option<String>,
    pub entry: Option<String>,
    pub format: Option<OutputFormat>,
    pub graph_options: GraphCompileOptions,
}

#[derive(Debug, Error)]
pub enum CompilePipelineError {
    #[error("failed to parse source: {0}")]
    Parse(#[from] metacat::theory::ast::ParseRawError),
    #[error("failed to elaborate source: {0}")]
    Elaborate(#[from] ElaborateError),
    #[error("failed to typecheck source: {0}")]
    Check(#[from] CheckError),
    #[error("failed to build compile graph: {0}")]
    CompileGraph(#[from] CompileGraphError),
    #[error("failed to render compile graph: {0}")]
    RenderGraph(#[from] std::io::Error),
    #[error(transparent)]
    Cuda(#[from] CudaCompileError),
    #[error("{argument} is required when emitting {emit:?}")]
    MissingArgument { argument: &'static str, emit: Emit },
    #[error("--format {format:?} is not supported when emitting {emit:?}")]
    UnsupportedFormat { emit: Emit, format: OutputFormat },
    #[error("--no-inline is only supported for emits that build a compile graph, not {0:?}")]
    UnsupportedNoInline(Emit),
}

pub fn compile(request: CompileRequest) -> Result<Vec<u8>, CompilePipelineError> {
    let mut pipeline = CompilePipeline::new(request);
    pipeline.emit()
}

pub struct CompilePipeline {
    request: CompileRequest,
    elaborated: Option<RawTheorySet>,
    checked: Option<TheorySet>,
}

impl CompilePipeline {
    pub fn new(request: CompileRequest) -> Self {
        Self {
            request,
            elaborated: None,
            checked: None,
        }
    }

    pub fn emit(&mut self) -> Result<Vec<u8>, CompilePipelineError> {
        match self.request.emit {
            Emit::Elaborated => {
                self.require_format(OutputFormat::Text)?;
                self.reject_graph_options()?;
                Ok(self.elaborated()?.to_hexpr_text().into_bytes())
            }
            Emit::Checked => {
                self.require_format(OutputFormat::Text)?;
                self.reject_graph_options()?;
                Ok(check_summary(self.checked()?).into_bytes())
            }
            Emit::CompileGraph => {
                self.require_format(OutputFormat::Svg)?;
                let graph = self.compile_graph()?;
                Ok(graph_render::nested_svg(&graph)?)
            }
            Emit::Cuda | Emit::StructuredIr => {
                self.require_format(OutputFormat::Text)?;
                let theory = self.required_input(PipelineInput::Theory)?;
                let entry = self.required_input(PipelineInput::Entry)?;
                let output = match self.request.emit {
                    Emit::Cuda => CudaOutput::Source,
                    Emit::StructuredIr => CudaOutput::StructuredIr,
                    _ => unreachable!("only CUDA-backed emits are handled here"),
                };
                let compile_graph = self.compile_graph()?;
                let checked = self.checked()?;
                Ok(
                    compile_cuda_from_graph(checked, &theory, &entry, &compile_graph, output)?
                        .into_bytes(),
                )
            }
        }
    }

    pub fn elaborated(&mut self) -> Result<&RawTheorySet, CompilePipelineError> {
        if self.elaborated.is_none() {
            let raw = RawTheorySet::from_files(self.request.paths.clone())?;
            self.elaborated = Some(elaborate(raw)?);
        }
        Ok(self.elaborated.as_ref().expect("elaborated is initialized"))
    }

    pub fn checked(&mut self) -> Result<&TheorySet, CompilePipelineError> {
        if self.checked.is_none() {
            let elaborated = self.elaborated()?;
            self.checked = Some(check_elaborated(elaborated)?);
        }
        Ok(self.checked.as_ref().expect("checked is initialized"))
    }

    pub fn compile_graph(&mut self) -> Result<CompileGraph, CompilePipelineError> {
        let theory = self.required_input(PipelineInput::Theory)?;
        let entry = self.required_input(PipelineInput::Entry)?;
        let graph_options = self.request.graph_options.clone();
        Ok(compile_graph_with_options(
            self.checked()?,
            &CompileConfig::data_control(),
            &theory,
            &entry,
            graph_options,
        )?)
    }

    fn required_input(&self, input: PipelineInput) -> Result<String, CompilePipelineError> {
        let value = match input {
            PipelineInput::Theory => self.request.theory.clone(),
            PipelineInput::Entry => self.request.entry.clone(),
        };
        value.ok_or(CompilePipelineError::MissingArgument {
            argument: input.name(),
            emit: self.request.emit,
        })
    }

    fn require_format(&self, expected: OutputFormat) -> Result<(), CompilePipelineError> {
        if let Some(format) = self.request.format
            && format != expected
        {
            return Err(CompilePipelineError::UnsupportedFormat {
                emit: self.request.emit,
                format,
            });
        }
        Ok(())
    }

    fn reject_graph_options(&self) -> Result<(), CompilePipelineError> {
        if !self.request.graph_options.no_inline.is_empty() {
            return Err(CompilePipelineError::UnsupportedNoInline(self.request.emit));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PipelineInput {
    Theory,
    Entry,
}

impl PipelineInput {
    fn name(self) -> &'static str {
        match self {
            PipelineInput::Theory => "theory",
            PipelineInput::Entry => "entry",
        }
    }
}

pub fn check_summary(theory_set: &TheorySet) -> String {
    let mut lines = vec!["OK: check passed".to_string()];
    for (id, theory) in &theory_set.theories {
        if let metacat::theory::Theory::Theory { arrows, .. } = theory {
            let definitions = arrows
                .values()
                .filter(|arrow| arrow.definition.is_some())
                .count();
            lines.push(format!("  {id}: {definitions} definitions"));
        }
    }
    lines.push(String::new());
    lines.join("\n")
}
