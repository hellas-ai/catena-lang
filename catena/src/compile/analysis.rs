use crate::compile::{CompileGraph, graph_render};

pub fn render_analysis(graph: &CompileGraph) -> std::io::Result<Vec<u8>> {
    render_step(AnalysisStep::NormalizedGraph, graph)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnalysisStep {
    NormalizedGraph,
}

fn render_step(step: AnalysisStep, graph: &CompileGraph) -> std::io::Result<Vec<u8>> {
    match step {
        AnalysisStep::NormalizedGraph => graph_render::nested_svg(graph),
    }
}
