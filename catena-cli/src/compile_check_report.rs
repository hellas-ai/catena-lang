use catena::compile::{ArrowType, CompileCheckReport};

use crate::hexpr_render;

pub fn print_compile_check_report(path: &str, report: &CompileCheckReport, verbose: bool) {
    println!("OK: compile check passed");
    println!("  file: {path}");
    for theory in &report.theories {
        println!(
            "  {}: {} definitions",
            theory.name, theory.report.definitions_checked
        );
    }
    for extension in &report.extensions {
        println!(
            "  lifted {} -> {}: {} arrows",
            extension.source,
            extension.target,
            extension.arrows.len()
        );
    }

    if verbose {
        for extension in &report.extensions {
            print_lift_report(
                &format!("{} -> {}", extension.source, extension.target),
                &extension.arrows,
            );
        }
    }
}

fn print_lift_report(label: &str, operations: &[ArrowType]) {
    println!("  {label}:");
    for arrow_type in operations {
        println!("    {}", hexpr_render::render_arrow_declaration(arrow_type));
    }
}
