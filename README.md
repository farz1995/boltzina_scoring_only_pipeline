# Boltzina Scoring-Only Pipeline

This repository is for running scoring-only affinity prediction from a crystal complex PDB using
`auto_scoring_pipeline.py`.

## Scope

This README only covers **scoring-only usage**.

- no docking is executed by default
- no AutoDock Vina is required for scoring mode
- input is a crystal complex PDB (`--complex-pdb`)

## What you need

- Python 3.10 to 3.12
- Linux shell (preferred) or Windows PowerShell
- Boltz-2 Python package and model files
- `maxit`, `pdb_chain`, `pdb_merge`, `pdb_tidy`, `pdb_rplresname`

## Install (once)

### 1) Python package and Boltz models

Linux / WSL:

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
python setup_boltzina.py
```

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install .
pip install rdkit-pypi
python setup_boltzina.py
```

### 2) External tools

Run the provided installer script if you are on Linux/WSL:

```bash
./setup.sh
```

That script installs MAXIT and a Vina binary. For scoring-only, Vina is optional and can be skipped.
What is required is `maxit` and the four pdb-tools executables.

If `pdb_chain`, `pdb_merge`, `pdb_tidy`, `pdb_rplresname` are missing, install them separately:

```bash
pip install pdb-tools
```

After install, make sure tools are on `PATH`.

Linux / WSL:

```bash
export RCSBROOT=<path-to-repo>/maxit-v11.300-prod-src
export PATH=$PATH:$RCSBROOT/bin
export PATH=$PATH:<path-to-repo>/bin
```

PowerShell:

```powershell
$env:RCSBROOT = "C:\path\to\boltzina\maxit-v11.300-prod-src"
$env:Path = "$env:Path;$env:RCSBROOT\bin;$PWD\bin"
```

## Run scoring pipeline

From repository root:

Linux / WSL:

```bash
python auto_scoring_pipeline.py --complex-pdb sample/1PYE.pdb
```

PowerShell:

```powershell
python auto_scoring_pipeline.py --complex-pdb .\sample\1PYE.pdb
```

Default output:

- `sample/<complex_stem>_auto_pipeline`

Generated in that folder:

- ligand pose file
- `<complex>_boltz_input.yaml`
- `<complex>_scoring_*.json`
- `pipeline_commands.txt`
- `results_scoring/` with result CSVs

## Common run options

Run Boltz + scoring (default):

```bash
python auto_scoring_pipeline.py --complex-pdb sample/1PYE.pdb
```

Use existing Boltz output (skip `boltz predict`):

```bash
python auto_scoring_pipeline.py `
  --complex-pdb sample/1PYE.pdb `
  --no-run-boltz `
  --work-dir sample/1PYE_boltz_results `
  --fname 1PYE_boltz_input
```

Dry run only:

```bash
python auto_scoring_pipeline.py --complex-pdb sample/1PYE.pdb --no-run-boltz --no-run-scoring --dry-run
```

Speed tuning:

```bash
python auto_scoring_pipeline.py --complex-pdb sample/1PYE.pdb --batch-size 2
```

## Validate CLI

```bash
python auto_scoring_pipeline.py --complex-pdb sample/1PYE.pdb --help
```

## Notes for a clean scoring-only install

- `bin/vina` and Vina-specific setup are optional for this mode.
- Keep only these essentials for scoring:
  - project source
  - `auto_scoring_pipeline.py`
  - Boltz install + model cache
  - external `maxit`/pdb-tools commands
- You can remove large generated artifacts (bytecode, build folders, and installer outputs) when not needed.
