#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from Bio.PDB import MMCIFParser, PDBParser, Superimposer, MMCIFIO, NeighborSearch
from Bio.PDB.SASA import ShrakeRupley
from collections import Counter


def load_af3_conf(conf_path: Path):
    with open(conf_path) as f:
        return json.load(f)


def get_atom_names(file_path: Path):
    if file_path.name.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    elif file_path.name.endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", file_path)

    atom_names = []
    for atom in structure.get_atoms():
        atom_names.append(atom.get_name())

    return atom_names


def backbone_plddt_stats(conf_json: dict, atom_names: list):
    """
    Returns:
      frac_bb_gt90
      mean_bb_plddt
    """
    atom_chain = np.array(conf_json["atom_chain_ids"])
    atom_plddt = np.array(conf_json["atom_plddts"])

    # protein backbone atoms
    mask = (
        (atom_chain == "A") &
        np.isin(atom_names, list({"N", "CA", "C", "O", "CB"}))
    )

    bb_vals = atom_plddt[mask]
    frac_gt90 = float(np.mean(bb_vals > 90.0))
    mean_bb = float(bb_vals.mean())

    return frac_gt90, mean_bb


def ligand_mean_plddt_stats(conf_json: dict):
    """
    Mean pLDDT for ligand atoms (chain B)
    """
    atom_chain = np.array(conf_json["atom_chain_ids"])
    atom_plddt = np.array(conf_json["atom_plddts"])

    lig_vals = atom_plddt[atom_chain == "B"]

    if len(lig_vals) == 0:
        return None

    return float(lig_vals.mean())


def compute_holo_sasa_metrics(
    holo_cif: Path,
    ligand_chain: str = "B",
    protein_chain: str = "A",
    n_points: int = 100,
    exposed_atom_thresh: float = 1.0,
    pocket_cutoff: float = 5.0,
):
    """
    Shrake-Rupley solvent-accessible surface area (SASA) analysis of an AF3
    holo complex. Used to flag designs whose binding pocket is so tightly
    wrapped around the cofactor / transition state that there is no opening for
    the cofactor and substrate to enter (such designs tend to score very well on
    pure confidence metrics, hence the trade-off this screen tries to expose).

    Returns a dict with:
      ligand_sasa_complex    : ligand SASA within the protein-ligand complex (A^2)
      ligand_sasa_free       : ligand SASA in isolation, same conformation (A^2)
      ligand_rel_sasa        : ligand_sasa_complex / ligand_sasa_free, i.e. the
                               fraction of the ligand surface exposed to solvent.
                               ~0 = fully buried / closed pocket (no entry route);
                               higher = a more open, solvent-accessible pocket.
      ligand_buried_sasa     : ligand_sasa_free - ligand_sasa_complex (A^2), the
                               ligand surface area occluded by the protein.
      n_exposed_lig_atoms    : # ligand atoms with SASA > exposed_atom_thresh
      frac_exposed_lig_atoms : fraction of ligand atoms that are solvent-exposed
      n_pocket_residues      : # protein residues lining the pocket (any atom
                               within pocket_cutoff of the ligand)
      pocket_residue_sasa    : summed SASA of the pocket-lining residues (A^2),
                               a protein-surface measure of how open the pocket
                               mouth is.

    Returns None if the protein or ligand chain is missing.
    """
    parser = MMCIFParser(QUIET=True) if holo_cif.name.endswith(".cif") else PDBParser(QUIET=True)
    model = parser.get_structure("holo", holo_cif)[0]

    if protein_chain not in model or ligand_chain not in model:
        return None

    sr = ShrakeRupley(n_points=n_points)
    sr.compute(model, level="A")  # sets .sasa on every atom of the complex

    lig_atoms = list(model[ligand_chain].get_atoms())
    if not lig_atoms:
        return None
    ligand_sasa_complex = float(sum(a.sasa for a in lig_atoms))
    n_exposed = int(sum(1 for a in lig_atoms if a.sasa > exposed_atom_thresh))

    # protein residues lining the pocket: any atom within pocket_cutoff of ligand
    ns = NeighborSearch(list(model[protein_chain].get_atoms()))
    pocket_res_ids = set()
    for a in lig_atoms:
        for nb in ns.search(a.coord, pocket_cutoff):
            pocket_res_ids.add(nb.get_parent().get_full_id())
    pocket_residue_sasa = float(sum(
        at.sasa
        for res in model[protein_chain].get_residues() if res.get_full_id() in pocket_res_ids
        for at in res.get_atoms()
    ))

    # SASA of the ligand alone (same coordinates, protein removed)
    lig_only = model[ligand_chain].copy()
    sr.compute(lig_only, level="A")
    ligand_sasa_free = float(sum(a.sasa for a in lig_only.get_atoms()))

    rel = (ligand_sasa_complex / ligand_sasa_free) if ligand_sasa_free > 0 else None

    return {
        "ligand_sasa_complex": ligand_sasa_complex,
        "ligand_sasa_free": ligand_sasa_free,
        "ligand_rel_sasa": rel,
        "ligand_buried_sasa": ligand_sasa_free - ligand_sasa_complex,
        "n_exposed_lig_atoms": n_exposed,
        "frac_exposed_lig_atoms": n_exposed / len(lig_atoms),
        "n_pocket_residues": len(pocket_res_ids),
        "pocket_residue_sasa": pocket_residue_sasa,
    }


def load_mean_plddt_ipae_confidence(conf_json: dict, ligand: bool = False) -> Dict:

    atom_chain = np.array(conf_json["atom_chain_ids"])
    atom_plddt = np.array(conf_json["atom_plddts"])
    token_chain = np.array(conf_json["token_chain_ids"])
    pae = np.array(conf_json["pae"])

    protein_atoms = atom_chain == "A"
    mean_plddt = float(atom_plddt[protein_atoms].mean())

    mean_ipae = None
    if ligand and "B" in token_chain:
        prot_idx = np.where(token_chain == "A")[0]
        lig_idx = np.where(token_chain == "B")[0]
        if len(prot_idx) and len(lig_idx):
            mean_ipae = float(pae[np.ix_(prot_idx, lig_idx)].mean())

    return {
        "mean_plddt": mean_plddt,
        "mean_ipae": mean_ipae,
    }


def pass_summary_filters(
    summary_conf: dict,
    ptm_cut: float,
    iptm_cut: float,
    pair_iptm_cut: float,
    ipae_min_cut: float,
    max_disorder: float
):
    ok = True
    reasons = []

    if summary_conf["ptm"] < ptm_cut:
        ok = False
        reasons.append("low_ptm")

    if summary_conf["iptm"] and summary_conf["iptm"] < iptm_cut:
        ok = False
        reasons.append("low_iptm")

    # protein-ligand interface
    if summary_conf["iptm"] and summary_conf["chain_pair_iptm"][0][1] < pair_iptm_cut:
        ok = False
        reasons.append("weak_interface")

    if summary_conf["iptm"] and summary_conf["chain_pair_pae_min"][0][1] > ipae_min_cut:
        ok = False
        reasons.append("high_ipae_min")

    if summary_conf["fraction_disordered"] > max_disorder:
        ok = False
        reasons.append("too_disordered")

    if summary_conf["has_clash"] > 0:
        ok = False
        reasons.append("clash")

    if summary_conf["iptm"]:
        return {
            "pass": ok,
            "reasons": reasons,
            "ptm": summary_conf["ptm"],
            "iptm": summary_conf["iptm"],
            "pair_iptm": summary_conf["chain_pair_iptm"][0][1],
            "ipae_min": summary_conf["chain_pair_pae_min"][0][1],
            "fraction_disordered": summary_conf["fraction_disordered"]
        }
    else:
        return {
            "pass": ok,
            "reasons": reasons,
            "ptm": summary_conf["ptm"],
            "fraction_disordered": summary_conf["fraction_disordered"]
        }


def plddt_bins(values: List[float]) -> Dict:
    bins = {
        "<70": 0, "70–75": 0, "75–80": 0,
        "80–85": 0, "85–90": 0, ">90": 0
    }
    for v in values:
        if v < 70: bins["<70"] += 1
        elif v < 75: bins["70–75"] += 1
        elif v < 80: bins["75–80"] += 1
        elif v < 85: bins["80–85"] += 1
        elif v < 90: bins["85–90"] += 1
        else: bins[">90"] += 1

    return {
        "count": len(values),
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "bins": bins,
    }


def ipae_bins(values: List[float]) -> Dict:
    bins = {
        "0–1": 0, "1–2": 0, "2–3": 0,
        "3–4": 0, "4–5": 0, "5–6": 0,
        "6–7": 0, "7–8": 0, "8–9": 0, 
        "9–10": 0, ">10": 0
    }
    for v in values:
        if v < 1: bins["0–1"] += 1
        elif v < 2: bins["1–2"] += 1
        elif v < 3: bins["2–3"] += 1
        elif v < 4: bins["3–4"] += 1
        elif v < 5: bins["4–5"] += 1
        elif v < 6: bins["5–6"] += 1
        elif v < 7: bins["6–7"] += 1
        elif v < 8: bins["7–8"] += 1
        elif v < 9: bins["8–9"] += 1
        elif v < 10: bins["9–10"] += 1
        else: bins[">10"] += 1

    return {
        "count": len(values),
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "bins": bins,
    }


def ptm_bins(values: List[float]) -> Dict:
    bins = {
        "<0.70": 0, "0.70–0.75": 0, "0.75–0.80": 0,
        "0.80–0.85": 0, "0.85–0.90": 0, ">0.90": 0
    }
    for v in values:
        if v < 0.70: bins["<0.70"] += 1
        elif v < 0.75: bins["0.70–0.75"] += 1
        elif v < 0.80: bins["0.75–0.80"] += 1
        elif v < 0.85: bins["0.80–0.85"] += 1
        elif v < 0.90: bins["0.85–0.90"] += 1
        else: bins[">0.90"] += 1

    return {
        "count": len(values),
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "bins": bins,
    }


def value_bins(values: List[float], edges: List[float], fmt: str = "{:.2f}") -> Dict:
    """
    Generic histogram binning for arbitrary numeric metrics (e.g. SASA values).
    `edges` are the bin boundaries; the result has len(edges)+1 bins:
    (<edges[0]), [edges[i], edges[i+1]) ..., (>=edges[-1]). None values are
    dropped. Output matches the dict format consumed by plot_bar().
    """
    vals = [v for v in values if v is not None]
    labels = ["<" + fmt.format(edges[0])]
    for i in range(len(edges) - 1):
        labels.append(fmt.format(edges[i]) + "–" + fmt.format(edges[i + 1]))
    labels.append(">" + fmt.format(edges[-1]))

    bins = {lab: 0 for lab in labels}
    for v in vals:
        bins[labels[int(np.digitize(v, edges))]] += 1

    return {
        "count": len(vals),
        "mean": float(np.mean(vals)) if vals else None,
        "median": float(np.median(vals)) if vals else None,
        "bins": bins,
    }


def _clean_xy(x: List, y: List):
    """Drop pairs where either value is None (for scatter / correlation)."""
    xs, ys = [], []
    for a, b in zip(x, y):
        if a is not None and b is not None:
            xs.append(a)
            ys.append(b)
    return xs, ys


def plot_bar(stats: Dict, title: str, out_path: Path):
    labels = list(stats["bins"].keys())
    counts = list(stats["bins"].values())

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(labels, counts)

    for bar, val in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 str(val), ha="center", va="bottom", fontsize=9)

    plt.xlabel("Metric bin")
    plt.ylabel("Number of designs")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_scatter(x: List[float], y: List[float], x_label: str, 
                 y_label: str, title: str, out_path: Path):
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, alpha=0.6)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_scatter_colored(x: List[float], y: List[float], c: List[float],
                         x_label: str, y_label: str, c_label: str,
                         title: str, out_path: Path):
    plt.figure(figsize=(5.8, 5))
    sc = plt.scatter(x, y, c=c, cmap="viridis", alpha=0.7)
    plt.colorbar(sc, label=c_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def load_ca_atoms(file_path: Path):
    if file_path.name.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    elif file_path.name.endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", file_path)
    cas = []
    for model in structure:
        for chain in model:
            for res in chain:
                if "CA" in res:
                    cas.append(res["CA"])
    return cas, structure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-af3o", "--af3_output_roots", type=Path, nargs="*")
    ap.add_argument("-w", "--wildcard", type=str, default="*_model_*")
    ap.add_argument("--mean_plddt_cut", type=float, default=88.0)
    ap.add_argument("--ligand_mean_plddt_cut", type=float, default=84.5)
    ap.add_argument("--plddt_frac_gt90_cut", type=float, default=0.82)
    ap.add_argument("--mean_ipae_cut", type=float, default=3.0)
    ap.add_argument("--ptm_cut", type=float, default=0.86)
    ap.add_argument("--iptm_cut", type=float, default=0.88)
    ap.add_argument("--ipae_min_cut", type=float, default=1.30)
    ap.add_argument("--max_disorder", type=float, default=0.1)
    # --- solvent-accessibility / pocket-openness (Shrake-Rupley SASA) ---
    ap.add_argument("--no_sasa", action="store_true",
                    help="Skip the SASA / pocket-openness analysis (faster).")
    ap.add_argument("--sasa_n_points", type=int, default=100,
                    help="Shrake-Rupley sphere sampling points. Lower is faster "
                         "(~64 is a good speed/accuracy trade-off for screening).")
    ap.add_argument("--sasa_ligand_chain", type=str, default="B")
    ap.add_argument("--sasa_protein_chain", type=str, default="A")
    ap.add_argument("--sasa_exposed_atom_thresh", type=float, default=1.0,
                    help="A ligand atom counts as solvent-exposed if its SASA (A^2) "
                         "exceeds this value.")
    ap.add_argument("--sasa_pocket_cutoff", type=float, default=5.0,
                    help="Protein atoms within this distance (A) of the ligand define "
                         "the pocket-lining residues.")
    ap.add_argument("--ligand_rel_sasa_min", type=float, default=0.0,
                    help="Minimum relative ligand SASA (fraction of the ligand surface "
                         "exposed to solvent). Designs below this are rejected as too "
                         "tightly wrapped, i.e. no opening for cofactor/substrate entry. "
                         "Default 0.0 = OFF; inspect ligand_rel_sasa_distribution.png "
                         "then set e.g. 0.10-0.20 to actually screen.")
    ap.add_argument("--ligand_rel_sasa_max", type=float, default=1.0,
                    help="Maximum relative ligand SASA. Rejects designs where the ligand "
                         "is so exposed it is not really pocketed. Default 1.0 = OFF.")
    ap.add_argument("--frac_exposed_lig_atoms_min", type=float, default=0.0,
                    help="Minimum fraction of ligand atoms that touch solvent. Unlike "
                         "rel_sasa (a burial-depth measure dominated by how deep a large "
                         "ligand sits), this tracks whether part of the ligand reaches "
                         "bulk solvent, i.e. an actual entrance to the pocket. Default "
                         "0.0 = OFF; inspect frac_exposed_lig_atoms_distribution.png then "
                         "set e.g. 0.15-0.25.")
    ap.add_argument("--n_exposed_lig_atoms_min", type=int, default=0,
                    help="Minimum number of solvent-exposed ligand atoms (absolute "
                         "counterpart of --frac_exposed_lig_atoms_min). Default 0 = OFF.")
    # ap.add_argument("-rfd3o", "--rfd3_output_roots", type=Path)
    ap.add_argument("-o", "--analysis_output_root", type=Path, default="outputs/AlphaFold3_analysis")
    ap.add_argument("--no_copy_pdb", action="store_true")
    args = ap.parse_args()

    apo_mean_plddts, holo_mean_plddts, ligand_mean_plddts, apo_ptms, holo_ptms = [], [], [], [], []
    mean_ipaes, ipae_min_vals, iptms = [], [], []
    # solvent-accessibility / pocket-openness metrics (aligned with the lists above)
    ligand_rel_sasas, ligand_sasa_complexes, ligand_buried_sasas = [], [], []
    frac_exposed_lig_atoms_list, pocket_residue_sasas = [], []

    for af3_output_root in args.af3_output_roots:
        new_output_root = af3_output_root.parent / (af3_output_root.name + "_selected")
        new_output_root.mkdir(parents=True, exist_ok=True)

        # defaults so the pass/print path works even with --no_copy_pdb
        folded_designs = set()
        output_folder = new_output_root / Path("folded_0")
        if not args.no_copy_pdb:
            i = 0
            while os.path.isdir(output_folder):
                for folded_design in output_folder.glob(args.wildcard + "_id_*_*_model.cif"):
                    folded_designs.add(folded_design.name)
                i += 1
                output_folder = new_output_root / Path("folded_" + str(i))
            output_folder.mkdir(parents=True, exist_ok=True)

        for model_0_dir in sorted(af3_output_root.glob(args.wildcard + "_id_0_apo")):
            x = 0
            model_x_dir = model_0_dir

            if_one_design_all_states_pass_thresholds = False
            best_apo_plddt = 0
            best_holo_plddt = 0
            best_ipae = args.ipae_min_cut

            while os.path.isdir(model_x_dir):
                all_states_completed = True

                apo_conf = next(filter(lambda x: not x.name.endswith("_summary_confidences.json"), \
                                   model_x_dir.glob("*_confidences.json")), None)
                if not apo_conf:
                    all_states_completed = False

                holo_model_x_dir = model_x_dir.with_name(model_x_dir.name[:-3] + "holo")
                holo_conf = next(filter(lambda x: not x.name.endswith("_summary_confidences.json"), \
                                holo_model_x_dir.glob("*_confidences.json")), None) if holo_model_x_dir.exists() else None
                if not holo_conf:
                    all_states_completed = False

                if all_states_completed:
                    apo_conf_json = load_af3_conf(apo_conf)
                    apo_metrics = load_mean_plddt_ipae_confidence(apo_conf_json)
                    atom_names = get_atom_names(next(model_x_dir.glob("*.cif")))
                    apo_frac_gt90, apo_mean_bb = backbone_plddt_stats(apo_conf_json, atom_names)
                    apo_sum_conf = next(model_x_dir.glob("*_summary_confidences.json"), None)
                    apo_summary_confidences = pass_summary_filters(load_af3_conf(apo_sum_conf),
                                                                args.ptm_cut,
                                                                args.iptm_cut,
                                                                args.iptm_cut,
                                                                args.ipae_min_cut,
                                                                args.max_disorder
                                                              )
                
                    holo_conf_json = load_af3_conf(holo_conf)
                    holo_metrics = load_mean_plddt_ipae_confidence(holo_conf_json, ligand=True)
                    atom_names = get_atom_names(next(holo_model_x_dir.glob("*.cif")))
                    holo_frac_gt90, holo_mean_bb = backbone_plddt_stats(holo_conf_json, atom_names)
                    ligand_mean_plddt = ligand_mean_plddt_stats(holo_conf_json)
                    holo_sum_conf = next(holo_model_x_dir.glob("*_summary_confidences.json"), None)
                    holo_summary_confidences = pass_summary_filters(load_af3_conf(holo_sum_conf),
                                                                args.ptm_cut,
                                                                args.iptm_cut,
                                                                args.iptm_cut,
                                                                args.ipae_min_cut,
                                                                args.max_disorder
                                                              )

                    # solvent-accessibility / pocket-openness of the holo complex
                    holo_sasa = None
                    if not args.no_sasa:
                        holo_cif_path = (next(holo_model_x_dir.glob("*_model.cif"), None)
                                         or next(holo_model_x_dir.glob("*.cif")))
                        holo_sasa = compute_holo_sasa_metrics(
                            holo_cif_path,
                            ligand_chain=args.sasa_ligand_chain,
                            protein_chain=args.sasa_protein_chain,
                            n_points=args.sasa_n_points,
                            exposed_atom_thresh=args.sasa_exposed_atom_thresh,
                            pocket_cutoff=args.sasa_pocket_cutoff,
                        )

                    apo_mean_plddts.append(apo_metrics["mean_plddt"])
                    apo_ptms.append(apo_summary_confidences["ptm"])
                    holo_mean_plddts.append(holo_metrics["mean_plddt"])
                    ligand_mean_plddts.append(ligand_mean_plddt)
                    holo_ptms.append(holo_summary_confidences["ptm"])
                    mean_ipaes.append(holo_metrics["mean_ipae"])
                    ipae_min_vals.append(holo_summary_confidences["ipae_min"])
                    iptms.append(holo_summary_confidences["iptm"])
                    ligand_rel_sasas.append(holo_sasa["ligand_rel_sasa"] if holo_sasa else None)
                    ligand_sasa_complexes.append(holo_sasa["ligand_sasa_complex"] if holo_sasa else None)
                    ligand_buried_sasas.append(holo_sasa["ligand_buried_sasa"] if holo_sasa else None)
                    frac_exposed_lig_atoms_list.append(holo_sasa["frac_exposed_lig_atoms"] if holo_sasa else None)
                    pocket_residue_sasas.append(holo_sasa["pocket_residue_sasa"] if holo_sasa else None)

                    # pocket must be open enough for cofactor/substrate entry
                    sasa_ok = True
                    if not args.no_sasa:
                        rel = holo_sasa["ligand_rel_sasa"] if holo_sasa else None
                        sasa_ok = (rel is not None and
                                   args.ligand_rel_sasa_min <= rel <= args.ligand_rel_sasa_max and
                                   holo_sasa["frac_exposed_lig_atoms"] >= args.frac_exposed_lig_atoms_min and
                                   holo_sasa["n_exposed_lig_atoms"] >= args.n_exposed_lig_atoms_min)

                    # print(model_x_dir)
                    # print("apo_frac_gt90")
                    # print(apo_frac_gt90)
                    # if apo_frac_gt90 > args.plddt_frac_gt90_cut:
                    #     print(True)
                    # else:
                    #     print(False)
                    # print("apo_mean_plddt")
                    # print(apo_metrics["mean_plddt"])
                    # if apo_metrics["mean_plddt"] > args.mean_plddt_cut:
                    #     print(True)
                    # else:
                    #     print(False)
                    # print("apo_summary_confidences")
                    # print(apo_summary_confidences["pass"])
                    # print('apo_summary_conf["ptm"]')
                    # print(apo_summary_confidences["ptm"])
                    # print('apo_summary_conf["fraction_disordered"]')
                    # print(apo_summary_confidences["fraction_disordered"])
                    # print("holo_frac_gt90")
                    # print(holo_frac_gt90)
                    # if holo_frac_gt90 > args.plddt_frac_gt90_cut:
                    #     print(True)
                    # else:
                    #     print(False)
                    # print("holo_mean_plddt")
                    # print(holo_metrics["mean_plddt"])
                    # if holo_metrics["mean_plddt"] > args.mean_plddt_cut:
                    #     print(True)
                    # else:
                    #     print(False)
                    # print("ligand_mean_plddt")
                    # print(ligand_mean_plddt)
                    # if ligand_mean_plddt > args.ligand_mean_plddt_cut:
                    #     print(True)
                    # else:
                    #     print(False)
                    # print("ligand_mean_ipae")
                    # print(holo_metrics["mean_ipae"])
                    # if holo_metrics["mean_ipae"] < args.mean_ipae_cut:
                    #     print(True)
                    # else:
                    #     print(False)
                    # print("holo_summary_confidences")
                    # print(holo_summary_confidences["pass"])
                    # print('summary_conf["ptm"]')
                    # print(holo_summary_confidences["ptm"])
                    # print('summary_conf["iptm"]')
                    # print(holo_summary_confidences["iptm"])
                    # print('summary_conf["chain_pair_iptm"][0][1]')
                    # print(holo_summary_confidences["pair_iptm"])
                    # print('summary_conf["chain_pair_pae_min"][0][1]')
                    # print(holo_summary_confidences["ipae_min"])
                    # print('summary_conf["fraction_disordered"]')
                    # print(holo_summary_confidences["fraction_disordered"])

                    if apo_summary_confidences["pass"] and \
                            apo_summary_confidences["ptm"] > args.ptm_cut and \
                            apo_frac_gt90 > args.plddt_frac_gt90_cut and \
                            apo_metrics["mean_plddt"] > args.mean_plddt_cut and \
                            holo_summary_confidences["pass"] and \
                            holo_summary_confidences["ptm"] > args.ptm_cut and \
                            holo_summary_confidences["iptm"] > args.iptm_cut and \
                            holo_frac_gt90 > args.plddt_frac_gt90_cut and \
                            ligand_mean_plddt > args.ligand_mean_plddt_cut and \
                            holo_metrics["mean_plddt"] > args.mean_plddt_cut and \
                            holo_metrics["mean_ipae"] < args.mean_ipae_cut and \
                            sasa_ok and \
                            holo_summary_confidences["ipae_min"] < best_ipae:
                        if_one_design_all_states_pass_thresholds = True
                        best_apo_plddt = apo_metrics["mean_plddt"]
                        best_holo_plddt = holo_metrics["mean_plddt"]
                        best_ipae = holo_summary_confidences["ipae_min"]
                        best_sasa = holo_sasa
                        best_model_dir = model_x_dir
                        best_holo_model_dir = holo_model_x_dir

                # elif apo_conf:
                #     if apo_summary_confidences["pass"] and \
                #             apo_frac_gt90 > args.plddt_frac_gt90_cut and \
                #             apo_metrics["mean_plddt"] > best_apo_plddt and \
                #             apo_metrics["mean_plddt"] > args.mean_plddt_cut:
                #         best_apo_plddt = apo_metrics["mean_plddt"]
                #         best_model_dir = model_x_dir

                # elif holo_conf:
                #     if holo_summary_confidences["pass"] and \
                #             holo_summary_confidences["iptm"] > best_iptm and \
                #             holo_frac_gt90 > args.plddt_frac_gt90_cut and \
                #             ligand_mean_plddt > args.ligand_mean_plddt_cut and \
                #             holo_metrics["mean_plddt"] > args.mean_plddt_cut and \
                #             holo_metrics["mean_ipae"] < args.mean_ipae_cut and \
                #             holo_summary_confidences["ipae_min"] < args.ipae_min_cut:
                #         best_holo_plddt = holo_metrics["mean_plddt"]
                #         best_iptm = holo_summary_confidences["iptm"]
                #         best_holo_model_dir = holo_model_x_dir

                x += 1
                dir_name = str(model_0_dir.name)
                model_x_dir = af3_output_root / (dir_name[:-5] + str(x) + dir_name[-4:])

            if not if_one_design_all_states_pass_thresholds:
                continue

            if_not_yet_copied = False
            apo_cif = next(best_model_dir.glob("*.cif"))
            if not apo_cif.name in folded_designs:
                if_not_yet_copied = True
            holo_cif = next(best_holo_model_dir.glob("*.cif"))
            if not holo_cif.name in folded_designs:
                if_not_yet_copied = True

            rfd3_output_roots = af3_output_root.with_name(af3_output_root.name[:af3_output_root.name.rfind("_AlphaFold3")])
            rfd3_model_name = model_0_dir.name[:model_0_dir.name.rfind("_id_")]
            rfd3_model_orig_path = rfd3_output_roots / rfd3_model_name / (rfd3_model_name + ".cif")
            if os.path.isfile(rfd3_model_orig_path):
                print(rfd3_model_orig_path)
                ca_ref, ref_struct = load_ca_atoms(rfd3_model_orig_path)
                print(len(ca_ref))
                if not args.no_copy_pdb and if_not_yet_copied:
                    shutil.copy(rfd3_model_orig_path, output_folder / (rfd3_model_name + ".cif"))
            else:
                with open("copy_error.log", "a") as pf:
                    pf.write(str(rfd3_model_orig_path) + "\n")
                    pf.write(str(output_folder / (rfd3_model_name + ".cif")) + "\n")

            apo_cif = next(best_model_dir.glob("*.cif"))
            print(apo_cif)
            print(best_apo_plddt)
            if not args.no_copy_pdb and if_not_yet_copied:
                # Align
                ca_mob, mob_struct = load_ca_atoms(apo_cif)
                sup = Superimposer()
                sup.set_atoms(ca_ref[:len(ca_mob)], ca_mob)
                sup.apply(mob_struct.get_atoms())
                # Save aligned mobile
                io = MMCIFIO()
                io.set_structure(mob_struct)
                io.save(str(output_folder / apo_cif.name))

            holo_cif = next(best_holo_model_dir.glob("*.cif"))
            print(holo_cif)
            print(best_holo_plddt)
            if not args.no_copy_pdb and if_not_yet_copied:
                # Align
                ca_mob, mob_struct = load_ca_atoms(holo_cif)
                sup = Superimposer()
                sup.set_atoms(ca_ref[:len(ca_mob)], ca_mob)
                sup.apply(mob_struct.get_atoms())
                # Save aligned mobile
                io = MMCIFIO()
                io.set_structure(mob_struct)
                io.save(str(output_folder / holo_cif.name))

            if not args.no_sasa and best_sasa is not None:
                print(f"  pocket: ligand_rel_sasa={best_sasa['ligand_rel_sasa']:.3f} "
                      f"frac_exposed_lig_atoms={best_sasa['frac_exposed_lig_atoms']:.3f} "
                      f"exposed_lig_atoms={best_sasa['n_exposed_lig_atoms']} "
                      f"buried_sasa={best_sasa['ligand_buried_sasa']:.1f} "
                      f"pocket_residue_sasa={best_sasa['pocket_residue_sasa']:.1f}")


    results = {
        "n_points": len(apo_mean_plddts),
        "apo_mean_plddts": apo_mean_plddts,
        "apo_ptms": apo_ptms,
        "holo_mean_plddts": holo_mean_plddts,
        "holo_ptms": holo_ptms,
        "ligand_mean_plddts": ligand_mean_plddts,
        "mean_ipaes": mean_ipaes,
        "ipae_min_vals": ipae_min_vals,
        "iptms": iptms,
        "ligand_rel_sasas": ligand_rel_sasas,
        "ligand_sasa_complexes": ligand_sasa_complexes,
        "ligand_buried_sasas": ligand_buried_sasas,
        "frac_exposed_lig_atoms": frac_exposed_lig_atoms_list,
        "pocket_residue_sasas": pocket_residue_sasas,
    }

    args.analysis_output_root.mkdir(exist_ok=True)

    json_path = args.analysis_output_root / "af3_summary_metrics.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"[INFO] Wrote summary JSON to {json_path}")

    # solvent-accessibility / pocket-openness summary + distribution feedback
    rel_sasa_vals = [v for v in ligand_rel_sasas if v is not None]
    if not args.no_sasa and rel_sasa_vals:
        arr = np.array(rel_sasa_vals)
        print(f"[INFO] Ligand relative SASA over {len(arr)} folded designs: "
              f"min={arr.min():.3f} median={np.median(arr):.3f} "
              f"mean={arr.mean():.3f} max={arr.max():.3f}")
        if args.ligand_rel_sasa_min <= 0.0 and args.ligand_rel_sasa_max >= 1.0:
            for thr in (0.05, 0.10, 0.15, 0.20):
                print(f"[INFO]   {int((arr >= thr).sum()):>5d}/{len(arr)} designs have "
                      f"ligand_rel_sasa >= {thr:.2f}")
            print("[INFO] Pocket-openness filter is OFF (default). Inspect "
                  "ligand_rel_sasa_distribution.png and the rel_sasa_vs_* trade-off "
                  "plots, then re-run with --ligand_rel_sasa_min (e.g. 0.10-0.20) to "
                  "screen out tightly-wrapped pockets.")
        else:
            print(f"[INFO] Pocket-openness filter active: "
                  f"{args.ligand_rel_sasa_min} <= ligand_rel_sasa <= {args.ligand_rel_sasa_max}")

        # frac_exposed_lig_atoms tracks an actual entrance better than rel_sasa for
        # a large, deeply-bound cofactor (see frac_exposed_lig_atoms_distribution.png)
        fe = np.array([v for v in frac_exposed_lig_atoms_list if v is not None])
        if len(fe):
            print(f"[INFO] Fraction of exposed ligand atoms over {len(fe)} designs: "
                  f"min={fe.min():.3f} median={np.median(fe):.3f} "
                  f"mean={fe.mean():.3f} max={fe.max():.3f}")
            if args.frac_exposed_lig_atoms_min <= 0.0 and args.n_exposed_lig_atoms_min <= 0:
                for thr in (0.10, 0.15, 0.20, 0.25):
                    print(f"[INFO]   {int((fe >= thr).sum()):>5d}/{len(fe)} designs have "
                          f"frac_exposed_lig_atoms >= {thr:.2f}")
            else:
                print(f"[INFO] Entrance filter active: frac_exposed_lig_atoms >= "
                      f"{args.frac_exposed_lig_atoms_min}, n_exposed_lig_atoms >= "
                      f"{args.n_exposed_lig_atoms_min}")

    plot_bar(plddt_bins(apo_mean_plddts), "Apo average pLDDT distribution",
             args.analysis_output_root / "apo_plddt_distribution.png")

    plot_bar(ptm_bins(apo_ptms), "Apo pTM distribution",
             args.analysis_output_root / "apo_ptm_distribution.png")

    plot_bar(plddt_bins(holo_mean_plddts), "Holo average pLDDT distribution",
             args.analysis_output_root / "holo_plddt_distribution.png")

    plot_bar(ptm_bins(holo_ptms), "Holo pTM distribution",
             args.analysis_output_root / "holo_ptm_distribution.png")

    plot_bar(plddt_bins(ligand_mean_plddts), "Ligand average pLDDT distribution",
             args.analysis_output_root / "ligand_plddt_distribution.png")

    plot_bar(ipae_bins(mean_ipaes), "Average iPAE distribution",
             args.analysis_output_root / "average_ipae_distribution.png")

    plot_bar(ipae_bins(ipae_min_vals), "Min iPAE distribution",
             args.analysis_output_root / "ipae_min_distribution.png")

    plot_bar(ptm_bins(iptms), "Holo ipTM distribution",
             args.analysis_output_root / "holo_iptm_distribution.png")

    r = str(np.corrcoef(apo_mean_plddts, holo_mean_plddts)[0, 1])
    plot_scatter(apo_mean_plddts, holo_mean_plddts, "Apo average pLDDT", 
                 "Holo average pLDDT", "Apo vs Holo pLDDT correlation", 
                 args.analysis_output_root / str("apo_vs_holo_plddt_scatter_" + r + ".png"))
    r = str(np.corrcoef(apo_ptms, holo_ptms)[0, 1])
    plot_scatter(apo_ptms, holo_ptms, "Apo pTM", 
                 "Holo pTM", "Apo vs Holo pTM correlation", 
                 args.analysis_output_root / str("apo_vs_holo_ptm_scatter_" + r + ".png"))
    r = str(np.corrcoef(apo_mean_plddts, apo_ptms)[0, 1])
    plot_scatter(apo_mean_plddts, apo_ptms, "Apo average pLDDT", 
                 "Apo pTM", "Apo pLDDT vs pTM correlation", 
                 args.analysis_output_root / str("apo_plddt_vs_ptm_scatter_" + r + ".png"))
    r = str(np.corrcoef(holo_mean_plddts, holo_ptms)[0, 1])
    plot_scatter(holo_mean_plddts, holo_ptms, "Holo average pLDDT", 
                 "Holo pTM", "Holo pLDDT vs pTM correlation", 
                 args.analysis_output_root / str("holo_plddt_vs_ptm_scatter_" + r + ".png"))
    r = str(np.corrcoef(ligand_mean_plddts, mean_ipaes)[0, 1])
    plot_scatter(ligand_mean_plddts, mean_ipaes, "Ligand average pLDDT", 
                 "Ligand mean iPAE", "Ligand pLDDT vs mean iPAE correlation", 
                 args.analysis_output_root / str("ligand_plddt_vs_mean_ipae_scatter_" + r + ".png"))
    r = str(np.corrcoef(ligand_mean_plddts, ipae_min_vals)[0, 1])
    plot_scatter(ligand_mean_plddts, ipae_min_vals, "Ligand average pLDDT", 
                 "Ligand iPAE min", "Ligand pLDDT vs iPAE min correlation", 
                 args.analysis_output_root / str("ligand_plddt_vs_ipae_min_scatter_" + r + ".png"))
    r = str(np.corrcoef(ligand_mean_plddts, iptms)[0, 1])
    plot_scatter(ligand_mean_plddts, iptms, "Ligand average pLDDT", 
                 "Ligand ipTM", "Ligand pLDDT vs ipTM correlation", 
                 args.analysis_output_root / str("ligand_plddt_vs_iptm_scatter_" + r + ".png"))
    r = str(np.corrcoef(mean_ipaes, iptms)[0, 1])
    plot_scatter(mean_ipaes, iptms, "Ligand mean iPAE",
                 "Ligand ipTM", "Ligand mean iPAE vs ipTM correlation",
                 args.analysis_output_root / str("mean_ipae_vs_iptm_scatter_" + r + ".png"))

    # ---- solvent-accessibility / pocket-openness plots ----
    if rel_sasa_vals:
        plot_bar(value_bins(ligand_rel_sasas, [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50], "{:.2f}"),
                 "Ligand relative SASA (fraction exposed) distribution",
                 args.analysis_output_root / "ligand_rel_sasa_distribution.png")
        plot_bar(value_bins(ligand_sasa_complexes, [10, 25, 50, 75, 100, 150, 200], "{:.0f}"),
                 "Ligand SASA in complex distribution (A^2)",
                 args.analysis_output_root / "ligand_sasa_complex_distribution.png")
        plot_bar(value_bins(ligand_buried_sasas, [100, 200, 300, 400, 500, 600, 700, 800], "{:.0f}"),
                 "Ligand buried SASA distribution (A^2)",
                 args.analysis_output_root / "ligand_buried_sasa_distribution.png")
        plot_bar(value_bins(frac_exposed_lig_atoms_list, [0.05, 0.10, 0.20, 0.30, 0.40, 0.50], "{:.2f}"),
                 "Fraction of exposed ligand atoms distribution",
                 args.analysis_output_root / "frac_exposed_lig_atoms_distribution.png")
        plot_bar(value_bins(pocket_residue_sasas, [200, 300, 400, 500, 600, 700, 800], "{:.0f}"),
                 "Pocket-lining residue SASA distribution (A^2)",
                 args.analysis_output_root / "pocket_residue_sasa_distribution.png")

        # confidence vs pocket-openness trade-off scatters
        for yvals, ylabel, fname in [
            (ligand_mean_plddts, "Ligand average pLDDT", "rel_sasa_vs_ligand_plddt"),
            (mean_ipaes, "Ligand mean iPAE", "rel_sasa_vs_mean_ipae"),
            (iptms, "Holo ipTM", "rel_sasa_vs_iptm"),
            (holo_mean_plddts, "Holo average pLDDT", "rel_sasa_vs_holo_plddt"),
        ]:
            cx, cy = _clean_xy(ligand_rel_sasas, yvals)
            r = np.corrcoef(cx, cy)[0, 1] if len(cx) > 1 else float("nan")
            plot_scatter(cx, cy, "Ligand relative SASA (fraction exposed)", ylabel,
                         f"Pocket openness vs confidence (r={r:.2f})",
                         args.analysis_output_root / f"{fname}_{r:.3f}.png")

        # confidence plane (pLDDT vs iPAE) colored by pocket openness
        cx, cy, cc = [], [], []
        for a, b, c in zip(ligand_mean_plddts, mean_ipaes, ligand_rel_sasas):
            if a is not None and b is not None and c is not None:
                cx.append(a)
                cy.append(b)
                cc.append(c)
        if cx:
            plot_scatter_colored(
                cx, cy, cc,
                "Ligand average pLDDT", "Ligand mean iPAE", "Ligand rel. SASA",
                "Confidence plane colored by pocket openness",
                args.analysis_output_root / "ligand_plddt_ipae_colored_by_rel_sasa.png")


if __name__ == "__main__":
    main()
