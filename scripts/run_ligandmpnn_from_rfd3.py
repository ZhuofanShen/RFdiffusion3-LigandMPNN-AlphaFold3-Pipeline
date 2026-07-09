#!/usr/bin/env python3
import argparse
import gzip
import shutil
import subprocess
import json
import os
from pathlib import Path

import gemmi


# ===================== USER SETTINGS =====================

LIGANDMPNN = "/opt/LigandMPNN/run.py"
CHECKPOINT = "/opt/LigandMPNN/model_params/ligandmpnn_v_32_005_25.pt"

# =========================================================


def gunzip(src: Path, dst: Path):
    if dst.exists():
        return
    with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout)


def cif_to_pdb(cif: Path, pdb: Path):
    if pdb.exists():
        return
    st = gemmi.read_structure(str(cif))
    st.write_pdb(str(pdb))


def run_ligandmpnn(pdb, outdir, chains_to_design, fixed_residues, batch_size, env):
    if (outdir / "seqs").exists():
        print(f"[SKIP] LigandMPNN already done for {pdb.name}")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    log = outdir / "ligandmpnn.log"

    cmd = [
        "python", LIGANDMPNN,
        "--checkpoint_ligand_mpnn", CHECKPOINT,
        "--model_type", "ligand_mpnn",
        "--pdb_path", str(pdb),
        "--fixed_residues", " ".join(fixed_residues),
        "--bias_AA", "H:-10.0,C:-10.0,M:-2.0,A:-2.0",
        "--out_folder", str(outdir),
        "--save_stats", "1",
        "--zero_indexed", "1",
        "--number_of_batches", "1",
        "--batch_size", str(batch_size),
    ]

    if chains_to_design is not None:
        cmd.append("--chains_to_design")
        cmd.append(chains_to_design)

    with open(log, "w") as f:
        subprocess.check_call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--wildcard", type=str, default="*_model_*.cif.gz")
    ap.add_argument("-c", "--chains_to_design", type=str, nargs="*")
    ap.add_argument("-cat", "--catalytic_residues", type=str, nargs="*", default=list())
    ap.add_argument("-od", "--output_directory", type=Path)
    ap.add_argument("-s", "--batch_size", type=int, default=15)
    ap.add_argument("--gpu", type=str, default="0")
    args = ap.parse_args()

    catalytic_residues = args.catalytic_residues

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    for gz in sorted(args.output_directory.glob(args.wildcard)):
        base = gz.stem.replace(".cif", "")
        ligandmpnn_out_dir = args.output_directory / base
        ligandmpnn_out_dir.mkdir(exist_ok=True)
        cif = ligandmpnn_out_dir / f"{base}.cif"
        pdb = ligandmpnn_out_dir / f"{base}.pdb"
        
        print(f"\n=== Processing {base} ===")

        if gz.name.endswith(".cif.gz"):
            gunzip(gz, cif)
            cif_to_pdb(cif, pdb)
        elif gz.name.endswith(".cif"):
            cif_to_pdb(gz, pdb)
        else:
            os.rename(gz, pdb)

        chains_to_design = None
        if args.chains_to_design and len(args.chains_to_design) > 0:
            chains_to_design = ",".join(args.chains_to_design)

        mapping_file = args.output_directory / Path(base + ".json")
        if os.path.isfile(mapping_file):
            mapping = json.loads(mapping_file.read_text())["diffused_index_map"]
            new_catalytic_residues = list()
            for catalytic_residue in catalytic_residues:
                new_catalytic_residues.append(mapping[catalytic_residue])
        else:
            new_catalytic_residues = catalytic_residues

        run_ligandmpnn(pdb, ligandmpnn_out_dir, chains_to_design, new_catalytic_residues, args.batch_size, env)


if __name__ == "__main__":
    main()
