# Automated Scoring-Only Pipeline

`auto_scoring_pipeline.py` provides a single-command workflow that starts from a crystal complex PDB and prepares/runs Boltzina scoring-only mode.

By default, the script auto-detects the ligand residue from `HETATM`, infers ligand SMILES from the extracted ligand pose, and extracts receptor sequence from `ATOM` records.

## What It Automates

1. Extract ligand pose from complex (`HETATM`) by residue selection.
2. Rewrite ligand atom names to unique values (`C1`, `C2`, `N1`, ...).
3. Infer ligand SMILES from the extracted ligand pose.
4. Generate a Boltz YAML input file.
5. Run `boltz predict` by default.
6. Resolve Boltz `work_dir`, manifest `fname`, and receptor protein PDB.
7. Optionally generate `prepared_mols.pkl` from ligand pose.
8. Generate scoring config JSON for `run.py`.
9. Run `python run.py <config> --batch_size 1` by default.

## Required Tools

For scoring run path in Boltzina, these binaries should be on `PATH`:

- `maxit`
- `pdb_chain`
- `pdb_merge`
- `pdb_tidy`
- `pdb_rplresname`

If `--run-boltz` is enabled, `boltz` must also be on `PATH`.

## Quick Start

```powershell
Set-Location "F:\Code\boltzina\boltzina"
python auto_scoring_pipeline.py --complex-pdb "sample\1PYE.pdb"
```

This runs the full pipeline by default (`--run-boltz` and `--run-scoring` are enabled).

## Dry Run (Plan + File Generation)

```powershell
python auto_scoring_pipeline.py `
  --complex-pdb "sample\1PYE.pdb" `
  --project-dir "sample\1PYE_auto_dry" `
  --no-run-boltz `
  --no-run-scoring `
  --dry-run
```

## Notes

- The only required argument is `--complex-pdb`.
- By default outputs go to `sample/<complex_stem>_auto_pipeline`, so different input PDB files do not overwrite each other.
- Use `--ligand-resname`, `--ligand-chain`, or `--ligand-resseq` only when auto-detection picks the wrong ligand.
- Use `--ligand-smiles` only as a fallback override if SMILES inference fails for your structure.
- If your Boltz output already exists, pass `--work-dir` and skip `--run-boltz`.
- The generated scoring config is saved as `config_scoring_auto.json` under `--project-dir`.
- The executed/planned commands are also saved to `pipeline_commands.txt`.

