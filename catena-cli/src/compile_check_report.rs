use catena::compile::{ArrowType, CompileCheckReport};

use crate::hexpr_render;

pub fn print_compile_check_report(path: &str, report: &CompileCheckReport, verbose: bool) {
    println!("OK: compile check passed");
    println!("  file: {path}");
    println!("  data: {} definitions", report.data.definitions_checked);
    println!(
        "  control + lifted data: {} definitions",
        report.control_with_data.definitions_checked
    );
    println!(
        "  data + lifted control: {} definitions",
        report.data_with_control.definitions_checked
    );
    println!(
        "  lifted data -> control: {} arrows",
        report.data_to_control.len()
    );
    println!(
        "  lifted control -> data: {} arrows",
        report.control_to_data.len()
    );

    if verbose {
        print_lift_report("data -> control", &report.data_to_control);
        print_lift_report("control -> data", &report.control_to_data);
    }
}

fn print_lift_report(label: &str, operations: &[ArrowType]) {
    println!("  {label}:");
    for arrow_type in operations {
        println!("    {}", hexpr_render::render_arrow_declaration(arrow_type));
    }
}
