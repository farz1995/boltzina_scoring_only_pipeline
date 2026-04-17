# Automated Scoring-Only Pipeline

`auto_scoring_pipeline.py` now accepts multiple input complexes in one run.

Given one or more crystal complex PDB files, the script:

1. Extracts ligand pose from `HETATM` records.
2. Normalizes ligand atom names.
3. Infers ligand SMILES.
4. Builds Boltz YAML.
5. Runs `boltz predict` (if enabled).
6. Resolves manifest and receptor.
7. Generates scoring config for `run.py`.
8. Runs `python run.py <config> --batch_size 1` (if enabled).

## Required tools

`maxit`, `pdb_chain`, `pdb_merge`, `pdb_tidy`, `pdb_rplresname` must be on `PATH`.
If `--run-boltz` is enabled, `boltz` is also required.

## How to run

```bash
python auto_scoring_pipeline.py --complex-pdb sample/1PYE.pdb
```

For multiple inputs:

```bash
python auto_scoring_pipeline.py --complex-pdb sample/1PYE.pdb sample/1HVR.pdb sample/6MX3.pdb
```

## Useful options

- `--no-run-boltz`: skip Boltz prediction and use existing `--work-dir`.
- `--no-run-scoring`: skip scoring.
- `--dry-run`: write files and log only.
- `--batch-size`: scoring batch size.
- `--project-dir`: base output folder when using one complex; for multiple inputs this is used as parent folder and each complex is placed under `<project-dir>/<stem>_auto_pipeline`.

For multiple complexes, `--work-dir` and `--fname` are not used automatically per input in this workflow; use without those flags.
