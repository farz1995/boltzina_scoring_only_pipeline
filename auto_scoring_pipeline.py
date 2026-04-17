#!/usr/bin/env python3
"""Single-command scoring-only pipeline bootstrap for Boltzina.

This script automates the manual preparation steps for scoring-only mode from
one or more complex PDB files.
It can run in dry-run mode to only create files/plan commands, or execute
external commands (`boltz predict`, `python run.py ...`) when requested.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
}

SOLVENT_RESNAMES = {"HOH", "WAT", "DOD"}
COMMON_ION_RESNAMES = {
    "NA", "K", "CL", "CA", "MG", "ZN", "MN", "FE", "CU", "CO", "NI", "CD", "HG", "SR", "BA"
}


class PipelineError(RuntimeError):
    """Raised when pipeline preconditions are not satisfied."""


def _load_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return handle.readlines()


def _extract_chain_sequence_from_atom(lines: Sequence[str], chain_id: str) -> str:
    seen: OrderedDict[Tuple[str, str], str] = OrderedDict()
    for line in lines:
        if not line.startswith("ATOM"):
            continue
        if len(line) < 27:
            continue
        line_chain = line[21].strip() or "A"
        if line_chain != chain_id:
            continue
        resname = line[17:20].strip().upper()
        resseq = line[22:26].strip()
        key = (line_chain, resseq)
        if key not in seen:
            seen[key] = AA3_TO_1.get(resname, "X")
    sequence = "".join(seen.values())
    if not sequence:
        raise PipelineError(
            f"No ATOM residues found for chain '{chain_id}' in complex file."
        )
    return sequence


def _detect_primary_protein_chain(lines: Sequence[str]) -> str:
    counts: Dict[str, int] = {}
    for line in lines:
        if not line.startswith("ATOM") or len(line) < 22:
            continue
        ch = line[21].strip() or "A"
        counts[ch] = counts.get(ch, 0) + 1
    if not counts:
        raise PipelineError("No ATOM records found in complex PDB.")
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _select_ligand_atom_lines(
    lines: Sequence[str],
    ligand_resname: Optional[str],
    ligand_chain: Optional[str],
    ligand_resseq: Optional[str],

) -> Tuple[List[str], str, str, str]:
    grouped: Dict[Tuple[str, str, str], List[str]] = {}
    target_resname = ligand_resname.upper() if ligand_resname else None
    target_chain = ligand_chain.strip() if ligand_chain else None
    target_resseq = ligand_resseq.strip() if ligand_resseq else None

    for line in lines:
        if not line.startswith("HETATM"):
            continue
        if len(line) < 27:
            continue
        resname = line[17:20].strip().upper()
        chain = line[21].strip() or "A"
        resseq = line[22:26].strip()

        key = (resname, chain, resseq)
        grouped.setdefault(key, []).append(line)

    if not grouped:
        raise PipelineError("No HETATM records found in complex PDB.")

    if target_resname or target_chain or target_resseq:
        explicit = [
            (key, atoms)
            for key, atoms in grouped.items()
            if (target_resname is None or key[0] == target_resname)
            and (target_chain is None or key[1] == target_chain)
            and (target_resseq is None or key[2] == target_resseq)
        ]
        if not explicit:
            scope = []
            if target_resname is not None:
                scope.append(f"resname={target_resname}")
            if target_chain is not None:
                scope.append(f"chain={target_chain}")
            if target_resseq is not None:
                scope.append(f"resseq={target_resseq}")
            raise PipelineError(f"Could not find ligand atoms in complex ({', '.join(scope)}).")
        (resname, chain, resseq), atoms = max(explicit, key=lambda kv: len(kv[1]))
        return atoms, resname, chain, resseq

    filtered = [
        (key, atoms)
        for key, atoms in grouped.items()
        if key[0] not in SOLVENT_RESNAMES and key[0] not in COMMON_ION_RESNAMES
    ]
    candidates = filtered if filtered else list(grouped.items())
    (resname, chain, resseq), atoms = max(candidates, key=lambda kv: len(kv[1]))
    return atoms, resname, chain, resseq


def _format_atom_line(
    original: str,
    serial: int,
    atom_name: str,
    output_resname: str,
    output_chain: str,
) -> str:
    padded = original.rstrip("\n")
    if len(padded) < 80:
        padded = padded.ljust(80)
    chars = list(padded)

    serial_str = f"{serial:>5}"
    chars[6:11] = list(serial_str)

    atom_name_str = atom_name.rjust(4)
    chars[12:16] = list(atom_name_str)

    resname_str = output_resname[:3].ljust(3)
    chars[17:20] = list(resname_str)

    chars[21] = output_chain[:1]

    return "".join(chars) + "\n"


def _make_unique_atom_names(ligand_lines: Sequence[str]) -> List[str]:
    counters: Dict[str, int] = {}
    rendered: List[str] = []
    for idx, line in enumerate(ligand_lines, start=1):
        element = (line[76:78].strip() if len(line) >= 78 else "") or line[12:16].strip()[:1] or "X"
        element = element.upper()
        counters[element] = counters.get(element, 0) + 1
        atom_name = f"{element}{counters[element]}"
        if len(atom_name) > 4:
            atom_name = atom_name[:4]
        rendered.append(_format_atom_line(line, idx, atom_name, "UNL", "A"))
    return rendered


def _write_ligand_pose_from_complex(
    complex_pdb: Path,
    output_pdb: Path,
    ligand_resname: Optional[str],
    ligand_chain: Optional[str],
    ligand_resseq: Optional[str],
) -> Tuple[int, str, str, str]:
    lines = _load_lines(complex_pdb)
    ligand_atom_lines, selected_resname, selected_chain, selected_resseq = _select_ligand_atom_lines(
        lines, ligand_resname, ligand_chain, ligand_resseq
    )
    unique = _make_unique_atom_names(ligand_atom_lines)

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    with output_pdb.open("w", encoding="utf-8") as handle:
        for line in unique:
            handle.write(line)
        handle.write("END\n")
    return len(unique), selected_resname, selected_chain, selected_resseq


def _infer_smiles_from_pose(pose_pdb: Path) -> str:
    try:
        from rdkit import Chem  # type: ignore
    except Exception as exc:
        raise PipelineError(
            "RDKit is required to infer ligand SMILES from the input PDB."
        ) from exc

    mol = Chem.MolFromPDBFile(str(pose_pdb), removeHs=False, sanitize=False)
    if mol is None:
        raise PipelineError(f"Failed to parse extracted ligand pose for SMILES inference: {pose_pdb}")

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        # Fallback to partial sanitization to keep robust behavior for imperfect crystal files.
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES,
            )
        except Exception:
            pass

    try:
        mol_no_h = Chem.RemoveHs(mol, sanitize=False)
        smiles = Chem.MolToSmiles(mol_no_h, canonical=True)
    except Exception:
        smiles = ""

    if not smiles:
        try:
            smiles = Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            smiles = ""

    if not smiles:
        raise PipelineError(
            "Could not infer ligand SMILES from the extracted ligand pose. "
            "Provide `--ligand-smiles` as an override if needed."
        )

    return smiles


def _write_boltz_yaml(
    yaml_path: Path,
    protein_chain: str,
    protein_sequence: str,
    ligand_chain: str,
    ligand_smiles: str,
) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "version: 1\n"
        "sequences:\n"
        "- protein:\n"
        "    id:\n"
        f"    - {protein_chain}\n"
        f"    sequence: {protein_sequence}\n"
        "- ligand:\n"
        f"    id: {ligand_chain}\n"
        f"    smiles: {ligand_smiles}\n"
        "properties:\n"
        "- affinity:\n"
        f"    binder: {ligand_chain}\n"
    )
    yaml_path.write_text(text, encoding="utf-8")


def _resolve_work_dir(base_out_dir: Path, explicit_work_dir: Optional[Path]) -> Path:
    if explicit_work_dir:
        manifest = explicit_work_dir / "processed" / "manifest.json"
        if not manifest.exists():
            raise PipelineError(f"Provided work_dir has no manifest: {manifest}")
        return explicit_work_dir

    candidates = list(base_out_dir.rglob("processed/manifest.json"))
    if not candidates:
        raise PipelineError(
            "Could not locate any 'processed/manifest.json' under Boltz output root. "
            "Provide --work-dir explicitly or run with --run-boltz."
        )

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.parent.parent


def _read_manifest_record_id(work_dir: Path, explicit_fname: Optional[str]) -> str:
    if explicit_fname:
        return explicit_fname
    manifest_path = work_dir / "processed" / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = data.get("records") or []
    if not records:
        raise PipelineError(f"No records found in manifest: {manifest_path}")
    return records[0]["id"]


def _resolve_receptor_pdb(work_dir: Path, fname: str) -> Path:
    direct = work_dir / "predictions" / fname / f"{fname}_model_0_protein.pdb"
    if direct.exists():
        return direct

    pred_dir = work_dir / "predictions" / fname
    if pred_dir.exists():
        proteins = sorted(pred_dir.glob("*_protein.pdb"))
        if proteins:
            return proteins[0]

    raise PipelineError(
        f"Could not find Boltz receptor protein PDB for record '{fname}' under {work_dir}."
    )


def _write_placeholder_vina_config(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "center_x = 0\n"
        "center_y = 0\n"
        "center_z = 0\n"
        "size_x = 20\n"
        "size_y = 20\n"
        "size_z = 20\n"
        "num_modes = 1\n"
        "seed = 1\n"
        "cpu = 1\n",
        encoding="utf-8",
    )


def _write_scoring_config(
    config_path: Path,
    work_dir: Path,
    vina_config: Path,
    fname: str,
    input_ligand_name: str,
    output_dir: Path,
    receptor_pdb: Path,
    ligand_pose_path: Path,
    ligand_chain_id: str,
    prepared_mols_file: Optional[Path],
) -> None:
    cfg = {
        "work_dir": work_dir.as_posix(),
        "vina_config": vina_config.as_posix(),
        "fname": fname,
        "input_ligand_name": input_ligand_name,
        "ligand_chain_id": ligand_chain_id,
        "output_dir": output_dir.as_posix(),
        "scoring_only": True,
        "receptor_pdb": receptor_pdb.as_posix(),
        "ligand_files": [ligand_pose_path.as_posix()],
    }
    if prepared_mols_file:
        cfg["prepared_mols_file"] = prepared_mols_file.as_posix()

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg, indent=4), encoding="utf-8")


def _build_prepared_mols_pkl(pose_pdb: Path, out_pkl: Path) -> None:
    try:
        from rdkit import Chem  # type: ignore
        import pickle
    except Exception as exc:
        raise PipelineError(
            "RDKit is required to build prepared_mols_file. "
            "Install environment dependencies first."
        ) from exc

    mol = Chem.MolFromPDBFile(str(pose_pdb), removeHs=False, sanitize=False)
    if mol is None:
        raise PipelineError(f"RDKit failed to read ligand pose PDB: {pose_pdb}")

    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        atom_name = info.GetName().strip().upper()
        atom.SetProp("name", atom_name)

    payload = {pose_pdb.stem: mol}
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as handle:
        pickle.dump(payload, handle)


def _check_required_tools(run_boltz: bool) -> Dict[str, bool]:
    required = ["maxit", "pdb_chain", "pdb_merge", "pdb_tidy", "pdb_rplresname"]
    if run_boltz:
        required.append("boltz")

    status: Dict[str, bool] = {}
    for name in required:
        status[name] = shutil.which(name) is not None
    return status


def _run_cmd(cmd: List[str], dry_run: bool) -> None:
    print("$", " ".join(cmd))
    if dry_run:
        return

    env = os.environ.copy()
    if cmd and cmd[0] == "boltz":
        # Avoid common MKL/libgomp conflict in mixed HPC environments.
        if env.get("MKL_THREADING_LAYER", "").upper() in ("", "INTEL"):
            env["MKL_THREADING_LAYER"] = "GNU"

    subprocess.run(cmd, check=True, env=env)


def _write_command_log(path: Path, commands: Sequence[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for cmd in commands:
            handle.write(" ".join(cmd) + "\n")


def _write_receptor_from_complex(complex_pdb: Path, output_pdb: Path) -> int:
    """Extract ATOM records from complex as receptor fallback PDB."""
    lines = _load_lines(complex_pdb)
    atom_lines = [line for line in lines if line.startswith("ATOM")]
    if not atom_lines:
        raise PipelineError(f"No ATOM records found in complex for receptor fallback: {complex_pdb}")

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    with output_pdb.open("w", encoding="utf-8") as handle:
        for line in atom_lines:
            handle.write(line)
        handle.write("END\n")
    return len(atom_lines)


def _run_single_complex(
    complex_pdb: Path,
    project_dir: Path,
    args: argparse.Namespace,
    tool_status: Optional[Dict[str, bool]] = None,
) -> None:
    """Run the full auto-scoring preparation + scoring flow for one complex."""
    lines = _load_lines(complex_pdb)
    protein_chain = args.protein_chain or _detect_primary_protein_chain(lines)
    protein_sequence = args.protein_sequence or _extract_chain_sequence_from_atom(lines, protein_chain)

    ligand_tag = args.ligand_resname.upper() if args.ligand_resname else "AUTO"
    ligand_pose = project_dir / "input_pdbs_pose" / f"{complex_pdb.stem}_{ligand_tag}_pose.pdb"
    atom_count, selected_resname, selected_chain, selected_resseq = _write_ligand_pose_from_complex(
        complex_pdb=complex_pdb,
        output_pdb=ligand_pose,
        ligand_resname=args.ligand_resname,
        ligand_chain=args.ligand_chain,
        ligand_resseq=args.ligand_resseq,
    )

    ligand_smiles = args.ligand_smiles or _infer_smiles_from_pose(ligand_pose)

    fallback_receptor = project_dir / "receptor_from_complex.pdb"
    fallback_receptor_atoms = _write_receptor_from_complex(complex_pdb, fallback_receptor)

    yaml_path = project_dir / f"{complex_pdb.stem}_boltz_input.yaml"
    _write_boltz_yaml(
        yaml_path=yaml_path,
        protein_chain=protein_chain,
        protein_sequence=protein_sequence,
        ligand_chain=args.boltz_ligand_chain,
        ligand_smiles=ligand_smiles,
    )

    boltz_out_dir = Path(args.boltz_out_dir).resolve() if args.boltz_out_dir else project_dir
    commands: List[List[str]] = []

    if args.run_boltz:
        commands.append([
            "boltz", "predict", str(yaml_path), "--out_dir", str(boltz_out_dir), "--use_msa_server"
        ])

    if tool_status is None:
        tool_status = _check_required_tools(run_boltz=args.run_boltz)

    missing = [name for name, ok in tool_status.items() if not ok]
    if missing:
        print("Missing tools on PATH:", ", ".join(missing))
        if args.run_boltz and "boltz" in missing:
            raise PipelineError("Cannot run Boltz because `boltz` is not on PATH.")

    for cmd in commands:
        _run_cmd(cmd, dry_run=args.dry_run)

    explicit_work_dir = Path(args.work_dir).resolve() if args.work_dir else None
    work_dir = _resolve_work_dir(boltz_out_dir, explicit_work_dir)
    fname = _read_manifest_record_id(work_dir, args.fname)
    try:
        receptor_pdb = _resolve_receptor_pdb(work_dir, fname)
        receptor_source = "boltz_prediction"
    except PipelineError:
        receptor_pdb = fallback_receptor
        receptor_source = "complex_fallback"
        print(
            "Boltz predicted protein PDB was not found; "
            f"using receptor fallback extracted from complex: {receptor_pdb}"
        )

    prepared_mols_file: Optional[Path] = None
    if args.build_mol_pkl:
        prepared_mols_file = project_dir / "mols_pose.pkl"
        _build_prepared_mols_pkl(ligand_pose, prepared_mols_file)

    vina_config = project_dir / "vina_placeholder.txt"
    _write_placeholder_vina_config(vina_config)

    scoring_output_dir = project_dir / "results_scoring"
    config_path = project_dir / f"config_scoring_{complex_pdb.stem}.json"
    _write_scoring_config(
        config_path=config_path,
        work_dir=work_dir,
        vina_config=vina_config,
        fname=fname,
        input_ligand_name=args.input_ligand_name,
        output_dir=scoring_output_dir,
        receptor_pdb=receptor_pdb,
        ligand_pose_path=ligand_pose,
        ligand_chain_id=args.boltz_ligand_chain,
        prepared_mols_file=prepared_mols_file,
    )

    if args.run_scoring:
        commands.append([
            "python", "run.py", str(config_path), "--batch_size", str(args.batch_size)
        ])
        _run_cmd(commands[-1], dry_run=args.dry_run)

    _write_command_log(project_dir / "pipeline_commands.txt", commands)

    print("\nPipeline artifacts created:")
    print(f"  Input complex     : {complex_pdb}")
    print(f"  Selected ligand   : {selected_resname} chain {selected_chain} resseq {selected_resseq}")
    print(f"  Ligand SMILES     : {ligand_smiles}")
    print(f"  Ligand atoms      : {atom_count}")
    print(f"  Ligand pose PDB   : {ligand_pose}")
    print(f"  Boltz YAML        : {yaml_path}")
    print(f"  Work dir          : {work_dir}")
    print(f"  Receptor PDB      : {receptor_pdb}")
    print(f"  Receptor source   : {receptor_source}")
    print(f"  Receptor ATOM rows: {fallback_receptor_atoms}")
    if prepared_mols_file:
        print(f"  prepared_mols.pkl : {prepared_mols_file}")
    print(f"  Config JSON       : {config_path}")
    print(f"  Commands log      : {project_dir / 'pipeline_commands.txt'}")
    print("\nTool checks:")
    for name, ok in tool_status.items():
        print(f"  {name:14} {'OK' if ok else 'MISSING'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automate Boltzina scoring-only preparation from a complex PDB."
    )
    parser.add_argument(
        "--complex-pdb",
        nargs="+",
        required=True,
        help="Input crystal complex PDB paths. Example: --complex-pdb a.pdb b.pdb c.pdb"
    )
    parser.add_argument("--ligand-resname", default=None, help="Optional ligand residue name override in complex (e.g., PM1)")
    parser.add_argument("--ligand-chain", default=None, help="Optional ligand chain filter in complex")
    parser.add_argument("--ligand-resseq", default=None, help="Optional ligand residue sequence id filter")
    parser.add_argument("--ligand-smiles", default=None, help="Optional ligand SMILES override for Boltz YAML")
    parser.add_argument("--project-dir", default=None, help="Pipeline output root directory (default: sample/<complex_stem>_auto_pipeline)")
    parser.add_argument("--protein-chain", default=None, help="Protein chain id; auto-detects largest ATOM chain if omitted")
    parser.add_argument("--protein-sequence", default=None, help="Protein sequence; auto-extracted from ATOM if omitted")
    parser.add_argument("--boltz-ligand-chain", default="B", help="Ligand chain id used in Boltz YAML and config")
    parser.add_argument("--input-ligand-name", default="UNL", help="Ligand residue name expected by Boltzina")
    parser.add_argument("--work-dir", default=None, help="Existing Boltz work_dir; skip discovery when set")
    parser.add_argument("--fname", default=None, help="Boltz record id; defaults to manifest first record id")
    parser.add_argument(
        "--run-boltz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run `boltz predict` after writing YAML (default: true)",
    )
    parser.add_argument("--boltz-out-dir", default=None, help="Boltz --out_dir (default: <project-dir>)")
    parser.add_argument("--build-mol-pkl", action="store_true", help="Create prepared_mols.pkl from ligand pose")
    parser.add_argument(
        "--run-scoring",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run `python run.py <config>` at the end (default: true)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size used when running run.py")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and create files only")

    args = parser.parse_args()

    complex_pdbs = [Path(path).resolve() for path in args.complex_pdb]
    for complex_pdb in complex_pdbs:
        if not complex_pdb.exists():
            raise PipelineError(f"Complex PDB not found: {complex_pdb}")

    if len(complex_pdbs) > 1 and (args.work_dir is not None or args.fname is not None):
        raise PipelineError(
            "When providing multiple --complex-pdb files, omit --work-dir and --fname. "
            "These values can be defined per complex from each Boltz output."
        )

    explicit_tool_check = _check_required_tools(run_boltz=args.run_boltz)
    missing = [name for name, ok in explicit_tool_check.items() if not ok]
    if missing:
        print("Missing tools on PATH:", ", ".join(missing))
        if args.run_boltz and "boltz" in missing:
            raise PipelineError("Cannot run Boltz because `boltz` is not on PATH.")
    base_project_dir = Path(args.project_dir).resolve() if args.project_dir else None

    used_names: Dict[str, int] = {}
    for complex_pdb in complex_pdbs:
        if base_project_dir is not None:
            if len(complex_pdbs) == 1:
                project_dir = base_project_dir
            else:
                project_name = f"{complex_pdb.stem}_auto_pipeline"
                used_index = used_names.get(project_name, 0) + 1
                used_names[project_name] = used_index
                if used_index > 1:
                    project_name = f"{project_name}_{used_index}"
                project_dir = base_project_dir / project_name
        else:
            project_name = f"{complex_pdb.stem}_auto_pipeline"
            used_index = used_names.get(project_name, 0) + 1
            used_names[project_name] = used_index
            if used_index > 1:
                project_name = f"{project_name}_{used_index}"
            project_dir = Path("sample") / project_name

        _run_single_complex(complex_pdb, project_dir, args, tool_status=explicit_tool_check)


if __name__ == "__main__":
    main()

