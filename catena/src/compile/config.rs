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
                    tensor: "product",
                    unit: "unit",
                },
                TheoryExtension {
                    target: "data",
                    source: "control",
                    prefix: "control",
                    tensor: "coproduct",
                    unit: "unit",
                },
            ],
        }
    }

    pub fn extensions_for_target<'a>(
        &'a self,
        target: &'a str,
    ) -> impl Iterator<Item = &'a TheoryExtension> {
        self.extensions
            .iter()
            .filter(move |extension| extension.target == target)
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

    pub fn lifted_prefixes(&self) -> Vec<&'static str> {
        self.extensions
            .iter()
            .map(|extension| extension.prefix)
            .collect()
    }
}
