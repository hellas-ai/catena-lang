//! Callback aliases owned by the experimental backend. Closure conversion
//! uses this table only when experimental codegen is selected.

pub(crate) const CONVERTED_PRIMITIVES: &[(&str, &str)] = &[
    ("core.if", "core.ifc"),
    ("core.if_guarded", "core.if_guardedc"),
    ("core.fold.bounded", "core.fold.boundedc"),
    ("core.fold.trace", "core.fold.tracec"),
    ("core.fold.bounded_u64", "core.fold.bounded_u64c"),
    ("core.fold.trace_u64", "core.fold.trace_u64c"),
    ("unsafe.launch", "unsafe.launchc"),
    ("unsafe.launch_shared", "unsafe.launch_sharedc"),
];

/// Converted names distinguish the checker's environment-aware signatures.
/// The existing lowering handlers already consume environment/function pairs,
/// so dispatch uses the source name without changing any argument ports.
pub(crate) fn source_primitive(operation: &str) -> &str {
    CONVERTED_PRIMITIVES
        .iter()
        .find(|(_, converted)| operation == *converted)
        .map_or(operation, |(source, _)| *source)
}
