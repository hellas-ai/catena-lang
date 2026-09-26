use crate::codegen::lower_types::CType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::codegen) enum StorageType {
    F32,
    Bf16,
}

impl StorageType {
    pub(in crate::codegen) fn c_type(self) -> &'static str {
        match self {
            Self::F32 => "float",
            Self::Bf16 => "catena_bf16_t",
        }
    }

    pub(in crate::codegen) fn load(self, expression: &str) -> String {
        match self {
            Self::F32 => expression.to_owned(),
            Self::Bf16 => format!("catena_bf16_to_f32({expression})"),
        }
    }

    pub(in crate::codegen) fn store(self, expression: &str) -> String {
        match self {
            Self::F32 => expression.to_owned(),
            Self::Bf16 => format!("catena_bf16_from_f32({expression})"),
        }
    }

    pub(in crate::codegen) fn element_type(self) -> CType {
        match self {
            Self::F32 => CType::F32,
            Self::Bf16 => CType::BF16,
        }
    }
}

const HIP_GFX11_TARGETS: &str = "defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)";

pub(in crate::codegen) fn hip_gfx11_guard() -> String {
    format!("#if {HIP_GFX11_TARGETS}\n")
}

pub(in crate::codegen) fn hip_wmma_include_guard() -> String {
    format!(
        "#if !defined(__HIP_DEVICE_COMPILE__) || {HIP_GFX11_TARGETS}\n#include <rocwmma/rocwmma.hpp>\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_policy_keeps_f32_accumulation_conversions_explicit() {
        assert_eq!(StorageType::F32.load("value"), "value");
        assert_eq!(StorageType::F32.store("sum"), "sum");
        assert_eq!(StorageType::Bf16.load("value"), "catena_bf16_to_f32(value)");
        assert_eq!(StorageType::Bf16.store("sum"), "catena_bf16_from_f32(sum)");
    }

    #[test]
    fn capability_guards_cover_the_supported_gfx11_targets() {
        let guard = hip_gfx11_guard();
        for target in ["gfx1100", "gfx1103", "gfx1150", "gfx1153"] {
            assert!(guard.contains(target));
        }
        assert!(hip_wmma_include_guard().contains("#include <rocwmma/rocwmma.hpp>"));
    }
}
