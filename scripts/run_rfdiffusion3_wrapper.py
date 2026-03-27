#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Tuple, List


# ----------------------------
# JSON helpers
# ----------------------------

def replace_all_pdb_paths(obj: Any, new_pdb: str) -> Tuple[Any, int]:
    """
    Replace ALL string values ending with '.pdb' (case-insensitive)
    anywhere in a JSON-like object with `new_pdb`.

    Returns:
        (new_object, n_replacements)
    """
    n = 0

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            vv, dn = replace_all_pdb_paths(v, new_pdb)
            out[k] = vv
            n += dn
        return out, n

    if isinstance(obj, list):
        out_list = []
        for v in obj:
            vv, dn = replace_all_pdb_paths(v, new_pdb)
            out_list.append(vv)
            n += dn
        return out_list, n

    if isinstance(obj, str):
        if obj.strip().lower().endswith(".pdb"):
            print(f"[INFO] Found template PDB path in JSON: {obj} and replace with new PDB: {new_pdb}")
            return new_pdb, 1
        return obj, 0

    # numbers, bools, None
    return obj, 0


# ----------------------------
# RFdiffusion3 runner
# ----------------------------

def run_rfd3_one(json_path: Path,
                out_dir: Path,
                n_batches: int,
                diffusion_batch_size: int,
                skip_existing: bool,
                prevalidate_inputs: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rfd3", "design",
        f"inputs={str(json_path)}",
        f"out_dir={str(out_dir)}",
        f"skip_existing={str(skip_existing)}",
        f"prevalidate_inputs={str(prevalidate_inputs)}",
        f"n_batches={n_batches}",
        f"diffusion_batch_size={diffusion_batch_size}",
    ]
    print("[RUN]", " ".join(cmd))
    # subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser(
        description="RFdiffusion3 wrapper: single-PDB mode or folder mode"
    )
    ap.add_argument("-j", "--template_json", required=True, type=Path,
                    help="Template RFdiffusion3 JSON (e.g. /home/szf/rfd3-ligandmpnn-af3-pipeline/inputs/KH209_options_0.json)")
    ap.add_argument("-i", "--input_pdb", required=True, type=Path,
                    help="Either a PDB file OR a directory containing PDBs and to dump JSONs, \
                    e.g. /home/szf/rfd3-ligandmpnn-af3-pipeline/inputs/KH209_NHC_2-4_annulation_SRT-rot2_search_0")
    ap.add_argument("-o", "--outputs_root", required=True, type=Path,
                    help="Root outputs folder, \
                    e.g. /home/szf/rfd3-ligandmpnn-af3-pipeline/outputs/KH209_NHC_2-4_annulation_SRT-rot2_search_0")
    ap.add_argument("-b", "--n_batches", type=int, default=100)
    ap.add_argument("-n", "--diffusion_batch_size", type=int, default=1,
                    help="1 batch of 4 size -> default 4")
    ap.add_argument("--skip_existing", action="store_true", default=False)
    ap.add_argument("--prevalidate_inputs", action="store_true", default=True)
    ap.add_argument("--max_workers", type=int, default=1,
                    help="Parallel jobs. Use 1 unless you really want multiple GPUs.")
    args = ap.parse_args()

    template_json = args.template_json.resolve()
    input_pdb = args.input_pdb.resolve()
    outputs_root = args.outputs_root.resolve()

    Path(outputs_root).mkdir(parents=True, exist_ok=True)

    rfd3_json_file_name = template_json.stem

    # Load template JSON and locate template pdb path
    template = json.loads(template_json.read_text())

    # Normalize template pdb to absolute if it is relative (best-effort)
    # If template uses relative paths, keep old as-is for replacement; replacement uses absolute new path.
    # That's fine because we replace by key-hint too.
    print(f"[INFO] Template JSON: {template_json}")

    # ----------------------------
    # Mode A: input_pdb is a file
    # ----------------------------
    if input_pdb.is_file():
        if not input_pdb.name.lower().endswith(".pdb"):
            raise ValueError(f"--input_pdb is a file but not .pdb: {input_pdb}")

        # We still update the template in-memory to use this pdb (optional but consistent)
        updated, nrep = replace_all_pdb_paths(obj=template, new_pdb=str(pdb_path.resolve()))
        if nrep == 0:
            raise RuntimeError(
                "Failed to replace PDB path in template JSON (no replacements). "
                "Your template JSON may not contain a PDB path as a plain string."
            )

        # Write a resolved json next to template for reproducibility
        single_json = template_json.parent / f"{rfd3_json_file_name}_resolved.json"
        single_json.write_text(json.dumps(updated, indent=2))
        print(f"[INFO] Wrote resolved single-job JSON: {single_json} (replaced {nrep} fields)")

        run_rfd3_one(
            json_path=single_json,
            out_dir=outputs_root,
            n_batches=args.n_batches,
            diffusion_batch_size=args.diffusion_batch_size,
            skip_existing=args.skip_existing,
            prevalidate_inputs=args.prevalidate_inputs,
        )
        return

    # ----------------------------
    # Mode B: input_pdb is a directory
    # ----------------------------
    if not input_pdb.is_dir():
        raise ValueError(f"--input_pdb must be a file or directory. Got: {input_pdb}")

    grid_pdb_dir = input_pdb

    pdbs = sorted([p for p in grid_pdb_dir.glob("*.pdb") if p.is_file()])
    if not pdbs:
        raise RuntimeError(f"No .pdb files found in the directory: {grid_pdb_dir}")

    print(f"[INFO] Grid search mode: {len(pdbs)} PDBs found in {grid_pdb_dir}")
    print(f"[INFO] Input JSONs will be dumped to: {grid_pdb_dir}")

    jobs: List[Tuple[Path, Path]] = []

    # Dump input JSONs
    for pdb_path in pdbs:
        rfd3_grid_json_file_name = f"{pdb_path.stem}_{rfd3_json_file_name}"
        grid_json_path = grid_pdb_dir / f"{rfd3_grid_json_file_name}.json"

        updated, nrep = replace_all_pdb_paths(obj=template, new_pdb=str(pdb_path.resolve()))
        if nrep == 0:
            raise RuntimeError(
                f"Failed to replace PDB path for {pdb_path}. "
                "No replacements were made; check template JSON structure."
            )
        grid_json_path.write_text(json.dumps(updated, indent=2))
        print(f"[INFO] Wrote input JSON: {grid_json_path} (replaced {nrep} fields)")

        jobs.append((grid_json_path, outputs_root))

    # Run jobs (optionally parallel)
    if args.max_workers <= 1:
        for jpath, outputs_root in jobs:
            run_rfd3_one(
                json_path=jpath,
                out_dir=outputs_root,
                n_batches=args.n_batches,
                diffusion_batch_size=args.diffusion_batch_size,
                skip_existing=args.skip_existing,
                prevalidate_inputs=args.prevalidate_inputs,
            )
    else:
        print(f"[INFO] Running jobs in parallel with max_workers={args.max_workers}")
        with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
            futs = [
                ex.submit(
                    run_rfd3_one,
                    jpath, odir,
                    args.n_batches,
                    args.diffusion_batch_size,
                    args.skip_existing,
                    args.prevalidate_inputs
                )
                for jpath, odir in jobs
            ]
            for f in as_completed(futs):
                f.result()


if __name__ == "__main__":
    main()
