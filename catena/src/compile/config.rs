#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileConfig {
    pub syntax: &'static str,
    pub extensions: Vec<TheoryExtension>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TheoryExtension {
    pub target: &'static str,
    pub source: &'static str,
    pub prefix: &'static str,
    pub tensor: &'static str,
    pub unit: &'static str,
}

impl CompileConfig {
    pub fn data_control() -> Self {
        Self {
            syntax: "syntax",
            extensions: vec![
                TheoryExtension {
                    target: "control",
                    source: "data",
                    prefix: "data",
                    tensor: "*",
                    unit: "1",
                },
                TheoryExtension {
                    target: "data",
                    source: "control",
                    prefix: "control",
                    tensor: "+",
                    unit: "0",
                },
            ],
        }
    }

    pub fn extension_for_target_and_prefix(
        &self,
        target: &str,
        prefix: &str,
    ) -> Option<&TheoryExtension> {
        self.extensions
            .iter()
            .find(|extension| extension.target == target && extension.prefix == prefix)
    }
}
