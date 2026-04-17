# Boltzina (Scoring-Only)

This repository runs **scoring-only** Boltz-2 evaluation from crystal complex PDB files using `auto_scoring_pipeline.py`.

- Docking is skipped by default.
- One or more input PDB files are supported.
- The output is written separately for each complex.

## Quick install

1. Create and activate a conda environment
```
conda create -n boltzina-scoring python=3.11 -y
conda activate boltzina-scoring
```

2. Install the package
```
pip install .
```

3. Download Boltz model files
```
python setup_boltzina.py
```

4. Install Boltz-2 dependencies
```
./setup.sh
```


5. Install additional Python tools
```
pip install rdkit-pypi pdb-tools
```

6. Install/enable external Boltzina dependencies
- `maxit`
- `pdb_chain`, `pdb_merge`, `pdb_tidy`, `pdb_rplresname`

If these tools are not already on `PATH`, add their directories manually.

## Run scoring on one or more complexes

Basic usage:

```
python auto_scoring_pipeline.py --complex-pdb sample/1PYE.pdb sample/1HVR.pdb
```

`--complex-pdb` accepts one or more PDB paths.

### Optional flags

- Run Boltz structure generation and scoring (default)
```
python auto_scoring_pipeline.py --complex-pdb sample/1PYE.pdb sample/1HVR.pdb
```

- Skip Boltz prediction and use existing Boltz work directory (one run at a time)
```
python auto_scoring_pipeline.py \
  --complex-pdb sample/1PYE.pdb \
  --no-run-boltz \
  --work-dir sample/1PYE_boltz_results \
  --fname 1PYE_boltz_input
```

- Dry run (generate files and command log only)
```
python auto_scoring_pipeline.py \
  --complex-pdb sample/1PYE.pdb sample/1HVR.pdb \
  --no-run-boltz \
  --no-run-scoring \
  --dry-run
```

- Change batch size
```
python auto_scoring_pipeline.py --complex-pdb sample/1PYE.pdb --batch-size 2
```

### Output layout

- `sample/<complex_stem>_auto_pipeline/` when `--project-dir` is not set
- `--project-dir` (single input): uses this exact directory
- `--project-dir` + `<complex_stem>_auto_pipeline` (multiple inputs)

Each run creates:

- `<complex>_pose.pdb`
- `<complex>_boltz_input.yaml`
- `config_scoring_<complex>.json`
- `pipeline_commands.txt`
- `results_scoring/` (scored results)

## Important notes

- `--work-dir` and `--fname` are for single-complex workflows only.
- For multiple input PDB files, omit both and let each complex resolve its own `fname` from Boltz output.
- Keep paths short to avoid command-line parsing issues.

## Reference

Furui, K, & Ohue, M. Boltzina: Efficient and Accurate Virtual Screening via Docking-Guided Binding Prediction with Boltz-2. AI for Accelerated Materials Design - NeurIPS 2025.
https://openreview.net/forum?id=OwtEQsd2hN
