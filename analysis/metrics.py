"""Metrics."""

import os
import tree
import numpy as np
from typing import List, Optional
import multiprocessing as mp
import re
from pathlib import Path
import pandas as pd

from tmtools import tm_align
from Bio.PDB import PDBParser
from Bio.Align import PairwiseAligner

from openfold.np import residue_constants
from data import utils as du
from analysis.postprocess import Postprocess
from analysis.postprocess_utils import summarize_statistics
from data.residue_constants import restypes


CA_IDX = residue_constants.atom_order["CA"]

CA_VIOLATION_METRICS = [
    "ca_ca_bond_dev",
    "ca_ca_valid_percent",
    "ca_steric_clash_percent",
    "num_ca_steric_clashes",
]

EVAL_METRICS = [
    "peptide_tm_score",
    "peptide_rmsd",
    "peptide_aligned_rmsd",
    "flip_alignment",
    "sequence_recovery",
    "sequence_similarity",
]

ALL_METRICS = CA_VIOLATION_METRICS + EVAL_METRICS


def calc_tm_score(pos_1, pos_2, seq_1, seq_2):
    tm_results = tm_align(pos_1, pos_2, seq_1, seq_2)
    return tm_results.tm_norm_chain1, tm_results.tm_norm_chain2


def calc_aligned_rmsd(pos_1, pos_2):
    aligned_pos_1 = du.rigid_transform_3D(pos_1, pos_2)[0]
    return np.mean(np.linalg.norm(aligned_pos_1 - pos_2, axis=-1))


def calc_rmsd(pos_1, pos_2, flip_align=True):
    """
    Calculate RMSD. Consider both head-to-head and head-to-tail alignments and returns the minimum RMSD if flip_align is True.

    Returns:
        rmsd: Calculated minimum RMSD value.
        flip: Boolean flag indicating whether the head-to-tail alignment was used.
    """
    rmsd = np.mean(np.linalg.norm(pos_1 - pos_2, axis=-1))
    pos_1_reverse = np.flip(pos_1, axis=0)
    rmsd_reverse = np.mean(np.linalg.norm(pos_1_reverse - pos_2, axis=-1))
    if flip_align:
        flip = rmsd_reverse < rmsd
        rmsd = rmsd_reverse if flip else rmsd
    else:
        flip = False
    return rmsd, flip


def calc_seq_recovery(seq_1, seq_2, flip=False):
    seq_1 = np.flip(seq_1) if flip else seq_1
    aatype_match = seq_1 == seq_2
    recovery = np.sum(aatype_match) / len(aatype_match)
    return recovery


def calc_seq_similarity(seq_1, seq_2, flip=False, mode="local"):
    seq_1 = np.flip(seq_1) if flip else seq_1
    seq_1_str = "".join([restypes[i] for i in seq_1])
    seq_2_str = "".join([restypes[i] for i in seq_2])
    aligner = PairwiseAligner()
    aligner.mode = mode
    alignments = aligner.align(seq_1_str, seq_2_str)
    if len(alignments) > 0:
        alignment = alignments[0]
        score = alignment.score
        similarity = score / len(seq_1)
    else:
        similarity = 0
    return similarity


def protein_metrics(
    *,
    pdb_path,
    atom37_pos,
    aatype,
    gt_atom37_pos,
    gt_aatype,
    diffuse_mask,
    flip_align=True,
):

    atom37_mask = np.any(atom37_pos, axis=-1)

    # Geometry
    bb_mask = np.any(atom37_mask, axis=-1)
    ca_pos = atom37_pos[..., CA_IDX, :][bb_mask.astype(bool)]
    ca_ca_bond_dev, ca_ca_valid_percent = ca_ca_distance(ca_pos)
    num_ca_steric_clashes, ca_steric_clash_percent = ca_ca_clashes(ca_pos)

    # Eval
    bb_diffuse_mask = (diffuse_mask * bb_mask).astype(bool)
    unpad_gt_scaffold_pos = gt_atom37_pos[..., CA_IDX, :][bb_diffuse_mask]
    unpad_pred_scaffold_pos = atom37_pos[..., CA_IDX, :][bb_diffuse_mask]
    seq = du.aatype_to_seq(gt_aatype[bb_diffuse_mask])
    _, tm_score = calc_tm_score(
        unpad_pred_scaffold_pos, unpad_gt_scaffold_pos, seq, seq
    )
    peptide_rmsd, flip = calc_rmsd(
        unpad_pred_scaffold_pos, unpad_gt_scaffold_pos, flip_align=flip_align
    )
    peptide_aligned_rmsd = calc_aligned_rmsd(
        unpad_pred_scaffold_pos, unpad_gt_scaffold_pos
    )

    unpad_gt_aatype = gt_aatype[bb_diffuse_mask]
    unpad_pred_aatype = aatype[bb_diffuse_mask]
    sequence_recovery = calc_seq_recovery(unpad_pred_aatype, unpad_gt_aatype, flip)
    sequence_similarity = calc_seq_similarity(
        unpad_pred_aatype, unpad_gt_aatype, flip, mode="local"
    )

    metrics_dict = {
        "ca_ca_bond_dev": ca_ca_bond_dev,
        "ca_ca_valid_percent": ca_ca_valid_percent,
        "ca_steric_clash_percent": ca_steric_clash_percent,
        "num_ca_steric_clashes": num_ca_steric_clashes,
        "peptide_tm_score": tm_score,
        "peptide_rmsd": peptide_rmsd,
        "peptide_aligned_rmsd": peptide_aligned_rmsd,
        "flip_alignment": flip,
        "sequence_recovery": sequence_recovery,
        "sequence_similarity": sequence_similarity,
    }

    metrics_dict = tree.map_structure(lambda x: np.mean(x).item(), metrics_dict)
    return metrics_dict


def ca_ca_distance(ca_pos, tol=0.1):
    ca_bond_dists = np.linalg.norm(ca_pos - np.roll(ca_pos, 1, axis=0), axis=-1)[1:]
    ca_ca_dev = np.mean(np.abs(ca_bond_dists - residue_constants.ca_ca))
    ca_ca_valid = np.mean(ca_bond_dists < (residue_constants.ca_ca + tol))
    return ca_ca_dev, ca_ca_valid


def ca_ca_clashes(ca_pos, tol=1.5):
    ca_ca_dists2d = np.linalg.norm(ca_pos[:, None, :] - ca_pos[None, :, :], axis=-1)
    inter_dists = ca_ca_dists2d[np.where(np.triu(ca_ca_dists2d, k=0) > 0)]
    clashes = inter_dists < tol
    return np.sum(clashes), np.mean(clashes)


def postprocess_metric(
    pdb_file: str,
    ori_dir: str,
    out_dir: str,
    lig_chain_id: Optional[str],
    xml: Optional[str],
    amber_relax: bool = False,
    rosetta_relax: bool = False
):
    print(f"Running postprocessing protocol on {pdb_file}")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("", pdb_file)[0]
    all_chains = [chain.id for chain in structure.get_chains()]
    if lig_chain_id is not None:
        all_chains.remove(lig_chain_id)
    else:
        lig_chain_id = all_chains[0] if all_chains else None
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    try:
        postprocess = Postprocess(
            pdb_file=pdb_file,
            lig_chain_id=lig_chain_id,
            ori_dir=ori_dir,
            out_dir=out_dir,
            xml=xml,
            amber_relax=amber_relax,
            rosetta_relax=rosetta_relax
        )
        postprocess()
    except Exception as e:
        print(f"An error occurred when processing {pdb_file}: {e}")


def _extract_fields(p: str) -> dict:
    path = Path(p)
    parts = path.parts

    target_name_dir = None
    seq_id_dir = None
    root_dir_path: Optional[Path] = None

    if "postprocess_results" in parts:
        i = parts.index("postprocess_results")
        if i >= 2:
            target_name_dir = parts[i - 2]
            seq_id_dir = parts[i - 1]
            if i - 2 > 0:
                root_dir_path = Path(*parts[: i - 2]).resolve()
            else:
                root_dir_path = Path(".").resolve()

    stem = path.name
    m = re.match(r'^([^_]+)_(.+?)_sample_(\d+)(?:\.\w+)?$', stem)
    target_from_file = seq_from_file = None
    sample_id_val: Optional[int] = None
    if m:
        target_from_file, seq_from_file, sample_id_str = m.groups()
        sample_id_val = int(sample_id_str)

    target_name = target_name_dir or target_from_file
    seq_id = seq_id_dir or seq_from_file

    file_path = None
    if root_dir_path and target_name and seq_id and sample_id_val is not None:
        file_path = (
            root_dir_path
            / target_name
            / seq_id
            / f"{target_name}_{seq_id}_sample_{sample_id_val}_final.pdb"
        )

    return {
        "root_dir": str(root_dir_path) if root_dir_path else None,
        "target_name": target_name,
        "seq_id": seq_id,
        "sample_id": sample_id_val if sample_id_val is not None else pd.NA,
        "file_path": str(file_path) if file_path else None,
    }

def postprocess_metric_parallel(
    files: List[str],
    ori_dir: str,
    nproc: Optional[int] = max(os.cpu_count() - 1, 1),
    lig_chain_ids : Optional[List[Optional[str]]] = None,
    xml: Optional[str] = None,
    out_path: str = "./postprocess_results.csv",
    amber_relax: bool = False,
    rosetta_relax: bool = False,
    save_best: bool = False,
):
    if lig_chain_ids is not None:
        if len(lig_chain_ids) != len(files):
            raise ValueError("Length of files and lig_chain_ids must match.")
    else:
        lig_chain_ids = [None] * len(files)

    args_list = []
    score_files = set()
    for pdb_file, lig_chain_id in zip(files, lig_chain_ids):
        out_dir = os.path.dirname(pdb_file)
        args_list.append((pdb_file, ori_dir, out_dir, lig_chain_id, xml, amber_relax, rosetta_relax))
        score_files.add(os.path.join(out_dir, "postprocess_results", "rosetta_score.sc"))

    with mp.Pool(nproc) as pool:
        pool.starmap(postprocess_metric, args_list)

    missing = [s for s in score_files if not os.path.exists(s)]
    for s in missing:
        print(f"Warning: score file {s} does not exist.")
    score_files = [s for s in score_files if os.path.exists(s)]

    if not score_files:
        print("No score files found. Exiting.")
        return

    df = summarize_statistics(list(score_files))
    meta = pd.DataFrame([_extract_fields(str(p)) for p in df.index])
    df_renamed = df.rename(columns={
        "ddg_norepack": "ddg",
        "rmsd": "postprocess_rmsd",
    })
    out_df = pd.concat([meta.reset_index(drop=True), df_renamed.reset_index(drop=True)], axis=1)[
        ["target_name", "seq_id", "sample_id", "ddg", "postprocess_rmsd", "file_path"]
    ]
    out_df = out_df.sort_values(by=["target_name", "seq_id", "ddg"], ascending=[True, True, True]).reset_index(drop=True)
    if save_best:
        best_idx = out_df.groupby(["target_name", "seq_id"])["ddg"].idxmin()
        out_df = (
            out_df.loc[best_idx]
            .sort_values(by=["target_name", "seq_id"], ascending=[True, True])
            .reset_index(drop=True)
        )
    out_df.to_csv(out_path, index=False)
    print(f"Postprocessing metrics saved to {out_path}")
