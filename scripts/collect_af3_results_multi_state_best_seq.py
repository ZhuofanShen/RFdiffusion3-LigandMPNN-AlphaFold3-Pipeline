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
from Bio.PDB import MMCIFParser, PDBParser, Superimposer, MMCIFIO
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


def ligand_mean_plddt(conf_json: dict):
    """
    Mean pLDDT for ligand atoms (chain B)
    """
    atom_chain = np.array(conf_json["atom_chain_ids"])
    atom_plddt = np.array(conf_json["atom_plddts"])

    lig_vals = atom_plddt[atom_chain == "B"]

    if len(lig_vals) == 0:
        return None

    return float(lig_vals.mean())


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
    ap.add_argument("-l", "--ligand_cifs", type=Path, nargs="*")
    ap.add_argument("--mean_plddt_cut", type=float, default=88.0)
    ap.add_argument("--ligand_mean_plddt_cut", type=float, default=84.5)
    ap.add_argument("--plddt_frac_gt90_cut", type=float, default=0.82)
    ap.add_argument("--mean_ipae_cut", type=float, default=3.0)
    ap.add_argument("--ptm_cut", type=float, default=0.86)
    ap.add_argument("--iptm_cut", type=float, default=0.88)
    ap.add_argument("--ipae_min_cut", type=float, default=1.30)
    ap.add_argument("--max_disorder", type=float, default=0.1)
    # ap.add_argument("-rfd3o", "--rfd3_output_roots", type=Path)
    ap.add_argument("-o", "--analysis_output_root", type=Path, default="outputs/AlphaFold3_analysis")
    ap.add_argument("--no_copy_pdb", action="store_true")
    args = ap.parse_args()

    apo_mean_plddts, holo_mean_plddts, ligand_mean_plddts, apo_ptms, holo_ptms = [], [], [], [], []
    mean_ipaes, ipae_min_vals, iptms = [], [], []

    for af3_output_root in args.af3_output_roots:
        new_output_root = af3_output_root.parent / (af3_output_root.name + "_selected")
        new_output_root.mkdir(parents=True, exist_ok=True)

        if not args.no_copy_pdb:
            i = 0
            output_folder = new_output_root / Path("folded_0")
            folded_designs = set()
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
            best_holo_plddt_list = 0
            best_ipae = args.ipae_min_cut

            while os.path.isdir(model_x_dir):
                all_states_completed = True

                apo_conf = next(filter(lambda x: not x.name.endswith("_summary_confidences.json"), \
                                   model_x_dir.glob("*_confidences.json")), None)
                if not apo_conf:
                    all_states_completed = False

                holo_model_x_dirs = list()
                holo_confs = list()
                for ligand_cif in args.ligand_cifs:
                    holo_model_x_dir = model_x_dir.with_name(model_x_dir.name[:-3] + "holo_" + ligand_cif.stem)
                    holo_model_x_dirs.append(holo_model_x_dir)
                    holo_conf = next(filter(lambda x: not x.name.endswith("_summary_confidences.json"), \
                                    holo_model_x_dir.glob("*_confidences.json")), None) if holo_model_x_dir.exists() else None
                    holo_confs.append(holo_conf)
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

                    pass_holo_summary_confidences = True
                    holo_mean_plddt_list = list()
                    mean_ipae_list = list()
                    holo_frac_gt90_list = list()
                    ligand_mean_plddt_list = list()
                    ptm_list = list()
                    ipae_min_list = list()
                    iptm_list = list()

                    for holo_model_x_dir, holo_conf in zip(holo_model_x_dirs, holo_confs):
                        holo_conf_json = load_af3_conf(holo_conf)
                        holo_metrics = load_mean_plddt_ipae_confidence(holo_conf_json, ligand=True)
                        holo_mean_plddt_list.append(holo_metrics["mean_plddt"])
                        mean_ipae_list.append(holo_metrics["mean_ipae"])
                        atom_names = get_atom_names(next(holo_model_x_dir.glob("*.cif")))
                        holo_frac_gt90_list.append(backbone_plddt_stats(holo_conf_json, atom_names)[0])
                        ligand_mean_plddt_list.append(ligand_mean_plddt(holo_conf_json))
                        holo_sum_conf = next(holo_model_x_dir.glob("*_summary_confidences.json"), None)
                        holo_summary_confidences = pass_summary_filters(load_af3_conf(holo_sum_conf),
                                                                    args.ptm_cut,
                                                                    args.iptm_cut,
                                                                    args.iptm_cut,
                                                                    args.ipae_min_cut,
                                                                    args.max_disorder
                                                                )
                        if not holo_summary_confidences["pass"]:
                            pass_holo_summary_confidences = False
                        ptm_list.append(holo_summary_confidences["ptm"])
                        ipae_min_list.append(holo_summary_confidences["ipae_min"])
                        iptm_list.append(holo_summary_confidences["iptm"])

                    apo_mean_plddts.append(apo_metrics["mean_plddt"])
                    apo_ptms.append(apo_summary_confidences["ptm"])
                    holo_mean_plddts.append(holo_mean_plddt_list)
                    ligand_mean_plddts.append(ligand_mean_plddt_list)
                    holo_ptms.append(ptm_list)
                    mean_ipaes.append(mean_ipae_list)
                    ipae_min_vals.append(ipae_min_list)
                    iptms.append(iptm_list)

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
                            pass_holo_summary_confidences and \
                            min(ptm_list) > args.ptm_cut and \
                            min(iptm_list) > args.iptm_cut and \
                            min(holo_frac_gt90_list) > args.plddt_frac_gt90_cut and \
                            min(ligand_mean_plddt_list) > args.ligand_mean_plddt_cut and \
                            min(holo_mean_plddt_list) > args.mean_plddt_cut and \
                            max(mean_ipae_list) < args.mean_ipae_cut and \
                            max(ipae_min_list) < best_ipae:
                        if_one_design_all_states_pass_thresholds = True
                        best_apo_plddt = apo_metrics["mean_plddt"]
                        best_holo_plddt_list = holo_mean_plddt_list
                        best_ipae = max(ipae_min_list)
                        best_model_dir = model_x_dir
                        best_holo_model_dirs = holo_model_x_dirs

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

            if if_one_design_all_states_pass_thresholds:
            # if 
                apo_cif = next(best_model_dir.glob("*.cif"))
                if not apo_cif.name in folded_designs:
                    if_not_yet_copied = True
            # elif :
                for best_holo_model_dir in best_holo_model_dirs:
                    holo_cif = next(best_holo_model_dir.glob("*.cif"))
                    if not holo_cif.name in folded_designs:
                        if_not_yet_copied = True

            if not if_one_design_all_states_pass_thresholds: # best_apo_plddt > 0 and best_holo_plddt > 0:
                continue

            rfd3_output_roots = af3_output_root.with_name(af3_output_root.name[:af3_output_root.name.rfind("_AlphaFold3")])
            rfd3_model_name = model_0_dir.name[:model_0_dir.name.rfind("_id_")]
            rfd3_model_orig_path = rfd3_output_roots / rfd3_model_name / (rfd3_model_name + ".cif")
            if os.path.isfile(rfd3_model_orig_path):
                print(rfd3_model_orig_path)
                ca_ref, ref_struct = load_ca_atoms(rfd3_model_orig_path)
                print(len(ca_ref))
                if not args.no_copy_pdb:
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

            for i, ligand_cif in enumerate(args.ligand_cifs):
                best_holo_model_dir = best_model_dir.with_name(best_model_dir.name[:-3] + "holo_" + ligand_cif.stem)
                holo_cif = next(best_holo_model_dir.glob("*.cif"))
                print(holo_cif)
                print(best_holo_plddt_list[i])
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


    results = {
        "n_points": len(apo_mean_plddts),
        "apo_mean_plddts": apo_mean_plddts,
        "apo_ptms": apo_ptms,
        "holo_mean_plddts": holo_mean_plddts,
        "holo_ptms": holo_ptms,
        "ligand_mean_plddts": ligand_mean_plddts,
        "mean_ipaes": mean_ipaes,
        "ipae_min_vals": ipae_min_vals,
        "iptms": iptms
    }

    args.analysis_output_root.mkdir(exist_ok=True)

    json_path = args.analysis_output_root / "af3_summary_metrics.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"[INFO] Wrote summary JSON to {json_path}")

    holo_mean_plddts = np.array(holo_mean_plddts).T
    holo_ptms = np.array(holo_ptms).T
    ligand_mean_plddts = np.array(ligand_mean_plddts).T
    mean_ipaes = np.array(mean_ipaes).T
    ipae_min_vals = np.array(ipae_min_vals).T
    iptms = np.array(iptms).T

    plot_bar(plddt_bins(apo_mean_plddts), "Apo average pLDDT distribution",
             args.analysis_output_root / "apo_plddt_distribution.png")

    plot_bar(ptm_bins(apo_ptms), "Apo pTM distribution",
             args.analysis_output_root / "apo_ptm_distribution.png")

    for i_state, ligand_cif in enumerate(args.ligand_cifs):
        plot_bar(plddt_bins(list(holo_mean_plddts[i_state])), "Holo average pLDDT distribution",
                args.analysis_output_root / str("holo_plddt_distribution_" + ligand_cif.stem + ".png"))

        plot_bar(ptm_bins(list(holo_ptms[i_state])), "Holo pTM distribution",
                args.analysis_output_root / str("holo_ptm_distribution_" + ligand_cif.stem + ".png"))

        plot_bar(plddt_bins(list(ligand_mean_plddts[i_state])), "Ligand average pLDDT distribution",
                args.analysis_output_root / str("ligand_plddt_distribution_" + ligand_cif.stem + ".png"))

        plot_bar(ipae_bins(list(mean_ipaes[i_state])), "Average iPAE distribution",
                args.analysis_output_root / str("average_ipae_distribution_" + ligand_cif.stem + ".png"))

        plot_bar(ipae_bins(list(ipae_min_vals[i_state])), "Min iPAE distribution",
                args.analysis_output_root / str("ipae_min_distribution_" + ligand_cif.stem + ".png"))

        plot_bar(ptm_bins(list(iptms[i_state])), "Holo ipTM distribution",
                args.analysis_output_root / str("holo_iptm_distribution_" + ligand_cif.stem + ".png"))

    r = str(round(np.corrcoef(apo_mean_plddts, list(holo_mean_plddts[0]))[0, 1], 2))
    plot_scatter(apo_mean_plddts, holo_mean_plddts[0], "Apo average pLDDT", 
                 "Holo average pLDDT", "Apo vs Holo pLDDT correlation", 
                 args.analysis_output_root / str("apo_vs_holo_plddt_scatter_" + r + ".png"))
    r = str(round(np.corrcoef(apo_ptms, list(holo_ptms[0]))[0, 1], 2))
    plot_scatter(apo_ptms, holo_ptms[0], "Apo pTM", 
                 "Holo pTM", "Apo vs Holo pTM correlation", 
                 args.analysis_output_root / str("apo_vs_holo_ptm_scatter_" + r + ".png"))
    r = str(round(np.corrcoef(apo_mean_plddts, apo_ptms)[0, 1], 2))
    plot_scatter(apo_mean_plddts, apo_ptms, "Apo average pLDDT", 
                 "Apo pTM", "Apo pLDDT vs pTM correlation", 
                 args.analysis_output_root / str("apo_plddt_vs_ptm_scatter_" + r + ".png"))
    r = str(round(np.corrcoef(list(holo_mean_plddts[0]), list(holo_ptms[0]))[0, 1], 2))
    plot_scatter(list(holo_mean_plddts[0]), list(holo_ptms[0]), "Holo average pLDDT", 
                 "Holo pTM", "Holo pLDDT vs pTM correlation", 
                 args.analysis_output_root / str("holo_plddt_vs_ptm_scatter_" + r + ".png"))
    r = str(round(np.corrcoef(list(ligand_mean_plddts[0]), list(mean_ipaes[0]))[0, 1], 2))
    plot_scatter(list(ligand_mean_plddts[0]), list(mean_ipaes[0]), "Ligand average pLDDT", 
                 "Ligand mean iPAE", "Ligand pLDDT vs mean iPAE correlation", 
                 args.analysis_output_root / str("ligand_plddt_vs_mean_ipae_scatter_" + r + ".png"))
    r = str(round(np.corrcoef(list(ligand_mean_plddts[0]), list(ipae_min_vals[0]))[0, 1], 2))
    plot_scatter(list(ligand_mean_plddts[0]), list(ipae_min_vals[0]), "Ligand average pLDDT", 
                 "Ligand iPAE min", "Ligand pLDDT vs iPAE min correlation", 
                 args.analysis_output_root / str("ligand_plddt_vs_ipae_min_scatter_" + r + ".png"))
    r = str(round(np.corrcoef(list(ligand_mean_plddts[0]), list(iptms[0]))[0, 1], 2))
    plot_scatter(list(ligand_mean_plddts[0]), list(iptms[0]), "Ligand average pLDDT", 
                 "Ligand ipTM", "Ligand pLDDT vs ipTM correlation", 
                 args.analysis_output_root / str("ligand_plddt_vs_iptm_scatter_" + r + ".png"))
    r = str(round(np.corrcoef(list(mean_ipaes[0]), list(iptms[0]))[0, 1], 2))
    plot_scatter(list(mean_ipaes[0]), list(iptms[0]), "Ligand mean iPAE", 
                 "Ligand ipTM", "Ligand mean iPAE vs ipTM correlation", 
                 args.analysis_output_root / str("mean_ipae_vs_iptm_scatter_" + r + ".png"))


if __name__ == "__main__":
    main()
