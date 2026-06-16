use std::{collections::BTreeMap, fs, io, path::Path};

use metacat::theory::{RawTheorySet, ast::RawTheory};

use crate::report::CompileReport;

pub fn dump_elaboration(report: &CompileReport, root_dir: &Path) -> io::Result<()> {
    let Some(full_elaborated_theory) = &report.elaborated else {
        return Ok(());
    };

    fs::write(
        root_dir.join("elaborated.hex"),
        full_elaborated_theory.to_hexpr_text(),
    )?;

    let dir = root_dir.join("elaboration");
    fs::create_dir_all(&dir)?;
    fs::write(
        dir.join("generated.hex"),
        generated_elaboration(&report.raw_theories, full_elaborated_theory)?.to_hexpr_text(),
    )?;
    Ok(())
}

fn generated_elaboration(
    raw: &RawTheorySet,
    elaborated: &RawTheorySet,
) -> io::Result<RawTheorySet> {
    let baseline = raw
        .clone()
        .with_extensions()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let mut theories = BTreeMap::new();

    for (theory_name, elaborated_theory) in &elaborated.theories {
        let baseline_theory = baseline.theories.get(theory_name);
        let mut arrows = BTreeMap::new();

        for (arrow_name, arrow) in &elaborated_theory.arrows {
            let existed_before =
                baseline_theory.is_some_and(|theory| theory.arrows.contains_key(arrow_name));
            if !existed_before {
                arrows.insert(arrow_name.clone(), arrow.clone());
            }
        }

        if !arrows.is_empty() {
            theories.insert(
                theory_name.clone(),
                RawTheory {
                    name: elaborated_theory.name.clone(),
                    syntax_category: elaborated_theory.syntax_category.clone(),
                    arrows,
                },
            );
        }
    }

    Ok(RawTheorySet {
        theories,
        extensions: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use metacat::theory::RawTheorySet;

    use crate::elaborate::elaborate;

    #[test]
    fn generated_elaboration_file_contains_only_elaborated_arrows() {
        let raw = RawTheorySet::from_text(
            r#"
            (theory type nat {
              (arr 1 : 0 -> 1)
              (arr bool : 0 -> 1)
              (arr val : 1 -> 1)
            })

            (theory program type {
              (arr id_bool : (bool val) -> (bool val))
            })
            "#,
        )
        .expect("test theory should parse");

        let elaborated = elaborate(raw.clone()).expect("test theory should elaborate");
        let generated_raw =
            super::generated_elaboration(&raw, &elaborated).expect("generated elaboration");
        let program: hexpr::Operation = "program".parse().unwrap();
        let name_id_bool: hexpr::Operation = "name.id_bool".parse().unwrap();
        let id_bool: hexpr::Operation = "id_bool".parse().unwrap();
        let program_theory = generated_raw
            .theories
            .get(&program)
            .expect("program theory should exist");
        assert!(program_theory.arrows.contains_key(&name_id_bool));
        assert!(!program_theory.arrows.contains_key(&id_bool));
    }
}
