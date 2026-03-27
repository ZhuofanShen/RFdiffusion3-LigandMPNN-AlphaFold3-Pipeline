#!/usr/bin/env python3
import argparse
import json
import re
from typing import List, Dict, Optional
import os
from pathlib import Path
import shutil


# ================= USER CONFIG =================

MODEL_SEEDS = [2]
TOP_K = None
NUM_RECYCLES=1
NUM_SAMPLES=1
RUN_HOLO = True
RUN_APO = True

AF3_IMAGE = "alphafold3:modified"
AF3_MODELS = "/opt/alphafold3_weights"
AF3_DATABASES = "/home/public_databases"

# ==============================================


def parse_ligandmpnn_fasta(fa_path: Path, first_k: Optional[int] = None) -> List[Dict]:
    """
    Parse a LigandMPNN FASTA file and return the first k designed sequences
    (excluding the WT sequence).

    If first_k is None, return all designed sequences.
    """
    entries = []
    lines = fa_path.read_text().splitlines()
    i = 0

    while i < len(lines):
        if not lines[i].startswith(">"):
            i += 1
            continue

        header = lines[i]
        seq = lines[i + 1].strip()
        i += 2

        # Skip WT (WT header does NOT contain overall_confidence)
        if "overall_confidence" not in header:
            continue

        def grab(key):
            m = re.search(rf"{key}=([0-9.]+)", header)
            return float(m.group(1)) if m else 0.0

        idx_match = re.search(r"id=(\d+)", header)
        idx = int(idx_match.group(1)) if idx_match else -1

        entry = {
            "id": idx,
            "sequence": seq,
            "overall": grab("overall_confidence"),
            "ligand": grab("ligand_confidence"),
        }

        entries.append(entry)

        # Stop early once we have first k sequences
        if first_k is not None and len(entries) >= first_k:
            break

    return entries


def parse_bond_pairs(mapping: dict, items: List[str]):
    if len(items) % 2 != 0:
        raise ValueError("Input list length must be even (pairs of atoms).")

    parsed = []

    def parse_atom(s: str):
        # Expect format: ChainResid,AtomName  e.g. A63,SG
        try:
            left, atom = s.split(",")
        except ValueError:
            raise ValueError(f"Invalid atom spec '{s}', expected 'A63,SG'")

        m = re.match(r"^([A-Za-z])(\d+)$", left)
        if not m:
            raise ValueError(f"Invalid residue spec '{left}', expected e.g. A63")

        chain = m.group(1)
        resid = int(m.group(2))

        return [chain, resid, atom]

    for i in range(0, len(items), 2):
        cat_res_index, cat_res_atom = items[i].split(",")
        new_cat_res_index = mapping.get(cat_res_index)
        if new_cat_res_index:
            # print("Parsing catalytic residue " + cat_res_index + " to " + new_cat_res_index)
            a1 = parse_atom(new_cat_res_index + "," + cat_res_atom)
        else:
            a1 = parse_atom(items[i])
        a2 = parse_atom(items[i + 1])
        parsed.append([a1, a2])

    return parsed


def write_af3_json(out_json: Path, name: str, seq: str,
                   model_seeds: list,
                   ccd_codes=None,
                   user_ccd=None,
                   bond_pairs=None):

    payload = {
        "name": name,
        "modelSeeds": model_seeds,
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": seq,
                    "unpairedMsa": "",
                    "pairedMsa": "",
                    "templates": []
                }
            }
        ],
        "dialect": "alphafold3",
        "version": 4
    }

    if ccd_codes:
        payload["sequences"].append({
            "ligand": {"id": "B", "ccdCodes": ccd_codes}
        })
        if bond_pairs:
            payload["bondedAtomPairs"] = bond_pairs
        if user_ccd:
            payload["userCCDPath"] = user_ccd

    out_json.write_text(json.dumps(payload, indent=2))


def run_af3(base_path: Path, input_dir: Path, output_root: Path, cuda_visible_devices: str, \
            num_recycles: int, num_samples: int):

    # The true output directory passed to AF3 is the parent directory (i.e., af3_output_root)
    # AF3 will make the subdirectory af3_output_dir.name under af3_output_dir.parent (specified in .json)

    json_in = Path("/work") / input_dir.relative_to(base_path)
    out_in = Path("/work") / output_root.relative_to(base_path)

    cmd = [
        "docker", "run", "-it",
        "--gpus", f"device={cuda_visible_devices}",
        "--volume", f"{base_path}:/work",
        "--volume", f"{AF3_MODELS}:/root/models",
        "--volume", f"{AF3_DATABASES}:/root/public_databases",
        AF3_IMAGE,
        "python", "run_alphafold.py",
        "--input_dir", f"{json_in}",
        "--model_dir", "/root/models",
        "--norun_data_pipeline",
        "--output_dir", f"{out_in}",
        "--num_recycles", str(num_recycles),
        "--num_diffusion_samples", str(num_samples)
    ]
    print(" ".join(cmd))
    # subprocess.check_call(cmd)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--json_dump_path", type=Path)
    ap.add_argument("-rfd3o", "--rfd3_output_root", type=Path)
    ap.add_argument("-af3o", "--af3_output_root", type=Path)
    ap.add_argument("-w", "--wildcard", type=str, default="*_model_*")
    ap.add_argument("-l", "--ligand_cif", type=Path)
    ap.add_argument("-b", "--bond_pairs", type=str, nargs="*", \
                    help="i.e., A63,SG B1,CD A152,NZ B1,CZ")
    ap.add_argument("--gpu", type=str, default="0")
    args = ap.parse_args()

    base_path = args.json_dump_path.parent.parent
    assert base_path == args.rfd3_output_root.parent.parent
    if not args.af3_output_root:
        args.af3_output_root = args.rfd3_output_root.parent / Path(str(args.rfd3_output_root.name) + "_AlphaFold3")

    if os.path.isdir(args.json_dump_path):
        shutil.rmtree(args.json_dump_path)
    args.json_dump_path.mkdir(parents=True, exist_ok=True)
    if RUN_HOLO:
        shutil.copy(args.ligand_cif, args.json_dump_path / args.ligand_cif.name)

    for ligandmpnn_dir in sorted(p for p in args.rfd3_output_root.glob(args.wildcard) if p.is_dir()):
        print(f"Checking directory {ligandmpnn_dir.name}")
        seq_dir = ligandmpnn_dir / "seqs"
        fa_files = list(seq_dir.glob("*.fa"))
        if not fa_files:
            continue

        # print(f"\n=== AF3 validation: {ligandmpnn_dir.name} ===")

        entries = parse_ligandmpnn_fasta(fa_files[0], first_k=TOP_K)
        # if TOP_K:
        #     entries = sorted(entries, key=lambda x: x["overall"] + x["ligand"], reverse=True)[:TOP_K]

        if RUN_HOLO:
            ccd_codes = list()
            with open(args.ligand_cif, "r") as pf:
                for line in pf:
                    if line.startswith("_chem_comp.id"):
                        ccd_codes.append(line.strip("\n").strip(" ").split(" ")[-1])
            bond_pairs = None
            if args.bond_pairs:
                mapping_file = ligandmpnn_dir.with_suffix(".json")
                mapping = json.loads(mapping_file.read_text())["diffused_index_map"]
                bond_pairs = parse_bond_pairs(mapping, args.bond_pairs)

        for entry in entries:
            id = str(entry["id"])

            if RUN_HOLO:
                af3_output_dir = args.af3_output_root / f"{ligandmpnn_dir.name}_id_{id}_holo"
                if os.path.isdir(af3_output_dir):
                    if list(af3_output_dir.glob("*_confidences.json")):
                        print(f"[SKIP] AF3 already done: {af3_output_dir.name}")
                    else:
                        shutil.rmtree(af3_output_dir)
                else:
                    json_path = args.json_dump_path / f"{ligandmpnn_dir.name}_id_{id}_holo.json"
                    write_af3_json(
                        json_path, f"{ligandmpnn_dir.name}_id_{id}_holo", entry["sequence"],
                        MODEL_SEEDS, ccd_codes, args.ligand_cif.name, bond_pairs
                    )

            if RUN_APO:
                af3_output_dir = args.af3_output_root / f"{ligandmpnn_dir.name}_id_{id}_apo"
                if os.path.isdir(af3_output_dir):
                    if list(af3_output_dir.glob("*_confidences.json")):
                        print(f"[SKIP] AF3 already done: {af3_output_dir.name}")
                    else:
                        shutil.rmtree(af3_output_dir)
                else:
                    json_path = args.json_dump_path / f"{ligandmpnn_dir.name}_id_{id}_apo.json"
                    write_af3_json(
                        json_path, f"{ligandmpnn_dir.name}_id_{id}_apo", entry["sequence"],
                        MODEL_SEEDS
                    )
    run_af3(base_path, args.json_dump_path, args.af3_output_root, args.gpu, NUM_RECYCLES, NUM_SAMPLES)


if __name__ == "__main__":
    main()
