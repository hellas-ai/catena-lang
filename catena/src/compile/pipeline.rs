use std::path::PathBuf;

use metacat::theory::{RawTheorySet, TheorySet};
use thiserror::Error;

use crate::{
    check::{CheckError, check as check_elaborated_theory},
    compile::{
        CompileConfig, CompileGraph, CompileGraphError, GraphCompileOptions, check_render,
        compile_graph,
        cuda::{CudaAbiError, render_cuda_source},
        graph_render,
        normalize::{NormalizeGraphError, normalize_graph},
        program::{ProgramCompileError, compile_program_from_graph},
        structured::{StructuredCompileError, compile_structured_program},
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
    Normalize(#[from] NormalizeGraphError),
    #[error(transparent)]
    Program(#[from] ProgramCompileError),
    #[error(transparent)]
    Structured(#[from] StructuredCompileError),
    #[error(transparent)]
    CudaAbi(#[from] CudaAbiError),
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
    checked_elaborated_theory: Option<TheorySet>,
}

impl CompilePipeline {
    pub fn new(request: CompileRequest) -> Self {
        Self {
            request,
            elaborated: None,
            checked_elaborated_theory: None,
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
                Ok(check_render::summary(self.checked_elaborated_theory()?).into_bytes())
            }
            Emit::CompileGraph => {
                self.require_format(OutputFormat::Svg)?;
                let compile_graph_request = self.compile_graph_request()?;
                let checked_elaborated_theory = self.checked_elaborated_theory()?;
                let graph = Self::compile_graph(checked_elaborated_theory, compile_graph_request)?;
                Ok(graph_render::nested_svg(&graph)?)
            }
            Emit::Cuda | Emit::StructuredIr => {
                self.require_format(OutputFormat::Text)?;
                let emit = self.request.emit;
                let compile_graph_request = self.compile_graph_request()?;
                let checked_elaborated_theory = self.checked_elaborated_theory()?;
                let compile_graph =
                    Self::compile_graph(checked_elaborated_theory, compile_graph_request)?;
                let graph = normalize_graph(&compile_graph)?;
                let program = compile_program_from_graph(&graph)?;
                let structured = compile_structured_program(&program)?;
                Ok(match emit {
                    Emit::Cuda => {
                        render_cuda_source(checked_elaborated_theory, &program, &structured)?
                    }
                    Emit::StructuredIr => structured.render_ir(),
                    _ => unreachable!("only structured-backed emits are handled here"),
                }
                .into_bytes())
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

    pub fn checked_elaborated_theory(&mut self) -> Result<&TheorySet, CompilePipelineError> {
        if self.checked_elaborated_theory.is_none() {
            let elaborated = self.elaborated()?;
            self.checked_elaborated_theory = Some(check_elaborated_theory(elaborated)?);
        }
        Ok(self
            .checked_elaborated_theory
            .as_ref()
            .expect("checked elaborated theory is initialized"))
    }

    pub fn compile_graph_request(&self) -> Result<CompileGraphRequest, CompilePipelineError> {
        Ok(CompileGraphRequest {
            theory: self.required_input(PipelineInput::Theory)?,
            entry: self.required_input(PipelineInput::Entry)?,
            graph_options: self.request.graph_options.clone(),
        })
    }

    pub fn compile_graph(
        checked_elaborated_theory: &TheorySet,
        request: CompileGraphRequest,
    ) -> Result<CompileGraph, CompilePipelineError> {
        compile_graph_from_checked(checked_elaborated_theory, request)
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileGraphRequest {
    pub theory: String,
    pub entry: String,
    pub graph_options: GraphCompileOptions,
}

fn compile_graph_from_checked(
    checked_elaborated_theory: &TheorySet,
    request: CompileGraphRequest,
) -> Result<CompileGraph, CompilePipelineError> {
    Ok(compile_graph(
        checked_elaborated_theory,
        &CompileConfig::data_control(),
        &request.theory,
        &request.entry,
        request.graph_options,
    )?)
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
