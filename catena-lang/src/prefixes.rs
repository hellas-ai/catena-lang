pub(crate) const NAME_PREFIX: &str = "name.";
pub(crate) const PARTIAL_PREFIX: &str = "partial.";
pub(crate) const CONST_PREFIX: &str = "const.";
pub(crate) const CONST_U64_PREFIX: &str = "const.u64.";
pub(crate) const CONST_U32_PREFIX: &str = "const.u32.";
pub(crate) const GENERATED_OPERATION_PREFIX: &str = "catena.";
pub(crate) const GENERATED_VARIABLE_PREFIX: &str = "__catena_";
pub(crate) const GENERATED_CONTEXT_PREFIX: &str = "catena.context.";
pub(crate) const GENERATED_PARTIAL_PREFIX: &str = "catena.partial.";

#[cfg(test)]
mod tests {
    use hexpr::Hexpr;

    use super::{GENERATED_CONTEXT_PREFIX, GENERATED_PARTIAL_PREFIX};

    #[test]
    fn generated_operation_names_roundtrip_through_hexpr_text() {
        for name in [
            format!("{GENERATED_CONTEXT_PREFIX}closure.example.0"),
            format!("{GENERATED_PARTIAL_PREFIX}identity.0.partial.f.1"),
        ] {
            let expr = Hexpr::Operation(name.parse().unwrap());
            let reparsed = expr.to_string().parse::<Hexpr>().unwrap();
            assert_eq!(reparsed, expr);
        }
    }
}
