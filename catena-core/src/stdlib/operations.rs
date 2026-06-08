pub const MONOIDAL_STRUCTURE_OPERATIONS: &[&str] = &[
    "val.*.intro",
    "val.*.elim",
    "val.+.intro",
    "val.+.elim",
    "2.intro",
    "2.elim",
    "distl",
    "distr",
    "unitl.intro",
    "unitl.elim",
    "elim2",
];

pub const CONTROL_FLOW_ONLY_OPERATIONS: &[&str] = &["merge", "never"];

pub const INTERLEAVED_CONTROL_PREFIX: &str = "control.";
pub const INTERLEAVED_DATA_PREFIX: &str = "data.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationKind {
    Instruction,
    MonoidalStructure,
    ControlFlow,
    InterleavedControl,
    InterleavedData,
}

pub fn operation_kind(operation: &str) -> OperationKind {
    if operation.starts_with(INTERLEAVED_CONTROL_PREFIX) {
        return OperationKind::InterleavedControl;
    }
    if operation.starts_with(INTERLEAVED_DATA_PREFIX) {
        return OperationKind::InterleavedData;
    }

    if CONTROL_FLOW_ONLY_OPERATIONS.contains(&operation) {
        OperationKind::ControlFlow
    } else if MONOIDAL_STRUCTURE_OPERATIONS.contains(&operation) {
        OperationKind::MonoidalStructure
    } else {
        OperationKind::Instruction
    }
}

pub fn actual_operation_name(operation: &str) -> &str {
    operation
        .strip_prefix(INTERLEAVED_CONTROL_PREFIX)
        .or_else(|| operation.strip_prefix(INTERLEAVED_DATA_PREFIX))
        .unwrap_or(operation)
}

pub fn actual_operation_kind(operation: &str) -> OperationKind {
    operation_kind(actual_operation_name(operation))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_stdlib_operations() {
        assert_eq!(
            operation_kind("val.*.intro"),
            OperationKind::MonoidalStructure
        );
        assert_eq!(operation_kind("merge"), OperationKind::ControlFlow);
        assert_eq!(operation_kind("f32.add"), OperationKind::Instruction);
    }

    #[test]
    fn classifies_interleaved_operation_prefixes() {
        assert_eq!(
            operation_kind("control.val.+.elim"),
            OperationKind::InterleavedControl
        );
        assert_eq!(
            operation_kind("data.unitl.elim"),
            OperationKind::InterleavedData
        );
    }

    #[test]
    fn resolves_actual_operation_name() {
        assert_eq!(actual_operation_name("control.merge"), "merge");
        assert_eq!(actual_operation_name("data.f32.add"), "f32.add");
        assert_eq!(actual_operation_name("val.*.intro"), "val.*.intro");
    }

    #[test]
    fn classifies_actual_operation_after_stripping_interleave_prefix() {
        assert_eq!(
            actual_operation_kind("data.unitl.elim"),
            OperationKind::MonoidalStructure
        );
        assert_eq!(
            actual_operation_kind("control.merge"),
            OperationKind::ControlFlow
        );
        assert_eq!(
            actual_operation_kind("data.f32.add"),
            OperationKind::Instruction
        );
    }
}
