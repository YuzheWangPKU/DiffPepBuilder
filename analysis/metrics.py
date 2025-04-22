""" Metrics. """
import os
import tree
import numpy as np
from typing import List
import multiprocessing as mp

from tmtools import tm_align
from Bio.PDB import PDBParser
from Bio.Align import PairwiseAligner

from openfold.np import residue_constants
from data import utils as du
from analysis.redock import FlexPepDock
from analysis.relax_utils import redock_para, statistic_redock
from data.residue_constants import restypes


CA_IDX = residue_constants.atom_order['CA']

CPU_NUM = os.cpu_count()


CA_VIOLATION_METRICS = [
    'ca_ca_bond_dev',
    'ca_ca_valid_percent',
    'ca_steric_clash_percent',
    'num_ca_steric_clashes',
]

EVAL_METRICS = [
    'peptide_tm_score',
    'peptide_rmsd',
    'peptide_aligned_rmsd',
    'flip_alignment',
    'sequence_recovery',
    'sequence_similarity',
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
        flip = (rmsd_reverse < rmsd)
        rmsd = rmsd_reverse if flip else rmsd
    else:
        flip = False
    return rmsd, flip

def calc_seq_recovery(seq_1, seq_2, flip=False):
    seq_1 = np.flip(seq_1) if flip else seq_1
    aatype_match = (seq_1 == seq_2)
    recovery = np.sum(aatype_match) / len(aatype_match)
    return recovery

def calc_seq_similarity(seq_1, seq_2, flip=False, mode="local"):
    seq_1 = np.flip(seq_1) if flip else seq_1
    seq_1_str = ''.join([restypes[i] for i in seq_1])
    seq_2_str = ''.join([restypes[i] for i in seq_2])
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
        unpad_pred_scaffold_pos, unpad_gt_scaffold_pos, seq, seq)
    peptide_rmsd, flip = calc_rmsd(unpad_pred_scaffold_pos, unpad_gt_scaffold_pos, flip_align=flip_align)
    peptide_aligned_rmsd = calc_aligned_rmsd(unpad_pred_scaffold_pos, unpad_gt_scaffold_pos)
    
    unpad_gt_aatype = gt_aatype[bb_diffuse_mask]
    unpad_pred_aatype = aatype[bb_diffuse_mask]
    sequence_recovery = calc_seq_recovery(unpad_pred_aatype, unpad_gt_aatype, flip)
    sequence_similarity = calc_seq_similarity(unpad_pred_aatype, unpad_gt_aatype, flip, mode="local")

    metrics_dict = {
        'ca_ca_bond_dev': ca_ca_bond_dev,
        'ca_ca_valid_percent': ca_ca_valid_percent,
        'ca_steric_clash_percent': ca_steric_clash_percent,
        'num_ca_steric_clashes': num_ca_steric_clashes,
        'peptide_tm_score': tm_score,
        'peptide_rmsd': peptide_rmsd,
        'peptide_aligned_rmsd': peptide_aligned_rmsd,
        'flip_alignment': flip,
        'sequence_recovery': sequence_recovery,
        'sequence_similarity': sequence_similarity
    }

    metrics_dict = tree.map_structure(lambda x: np.mean(x).item(), metrics_dict)
    return metrics_dict 

def ca_ca_distance(ca_pos, tol=0.1):
    ca_bond_dists = np.linalg.norm(
        ca_pos - np.roll(ca_pos, 1, axis=0), axis=-1)[1:]
    ca_ca_dev = np.mean(np.abs(ca_bond_dists - residue_constants.ca_ca))
    ca_ca_valid = np.mean(ca_bond_dists < (residue_constants.ca_ca + tol))
    return ca_ca_dev, ca_ca_valid

def ca_ca_clashes(ca_pos, tol=1.5):
    ca_ca_dists2d = np.linalg.norm(
        ca_pos[:, None, :] - ca_pos[None, :, :], axis=-1)
    inter_dists = ca_ca_dists2d[np.where(np.triu(ca_ca_dists2d, k=0) > 0)]
    clashes = inter_dists < tol
    return np.sum(clashes), np.mean(clashes) 

def redock_metric(redock_para:redock_para):
    pdb_file = redock_para.test_file
    ori_path = redock_para.ori_path
    out_path = redock_para.out_path
    app = redock_para.app
    app_path = redock_para.app_path
    nproc = redock_para.nproc
    lig_ch = redock_para.lig_ch
    denovo = redock_para.denovo
    xml=redock_para.xml
    if app == 'flexpepdock' or 'ADCP':
        redock=True
        print(f'Running {app} redock protocol on {pdb_file}')
    else:
        redock=False
        print(f'Running interface analyze protocol on {pdb_file}')
    p = PDBParser(QUIET=1)
    s = p.get_structure('', pdb_file)[0]
    chains= [i.id for i in s.get_chains()]
    if lig_ch is not None:
        chains.remove(lig_ch)
        rec_chs = chains
    else:
        lig_ch = chains[0]
        rec_chs = chains[1:]
    fixed_pdbs_path = os.path.join(out_path, 'pdbs_fixed')
    relaxed_pdbs_path = os.path.join(out_path, 'pdbs_relaxed')
    redock_pdbs_path = os.path.join(out_path, 'pdbs_redock')
    rcon_pdbs_path = os.path.join(out_path, 'pdbs_rcon')
    if not os.path.exists(fixed_pdbs_path):
        os.makedirs(fixed_pdbs_path, exist_ok=True)
    if not os.path.exists(relaxed_pdbs_path):
        os.makedirs(relaxed_pdbs_path, exist_ok=True)
    if redock and (not os.path.exists(redock_pdbs_path)):
        os.makedirs(redock_pdbs_path, exist_ok=True)
    if not os.path.exists(rcon_pdbs_path):
        os.makedirs(rcon_pdbs_path, exist_ok=True)
    try:
        redock_object = FlexPepDock(pdb_file, rec_chs, lig_ch, rcon_pdbs_path,\
                                    ori_path, fixed_pdbs_path, relaxed_pdbs_path,\
                                    redock_pdbs_path, nproc=nproc, xml=xml)
        redock_object(app, app_path, denovo)
    except Exception as e:
        print(f"An error occurred when processing {pdb_file}: {e}")


def redock_metric_parallel(files:List[str], ori_path, nproc=CPU_NUM-1, app=None, app_path=None, lig_chs=None, denovo=False, xml=None, out_path='./'):

    if lig_chs is not None:
        assert len(files) == len(lig_chs)
    else:
        lig_chs = [None for i in files]
    
    _runs = [redock_para(i[0], ori_path, os.path.dirname(i[0]), app, app_path, nproc, i[1], i[2], i[3])\
              for i in zip(files, lig_chs, [denovo for j in files], [xml for k in files])]
    
    if app == 'ADCP':
        for run in _runs:
            redock_metric(run)
        docking_pathes = [os.path.join(os.path.dirname(i), 'pdbs_redock') for i in files]
        score_files = []
        for docking_path in docking_pathes:
            score_files.append(os.path.join(docking_path, os.path.splitext(os.path.basename(run.test_file))[0] + '_adcp','log.txt'))
        df =  statistic_redock(score_files, app=app, keys=None)
    else:
        pool = mp.Pool(nproc)
        pool.map(redock_metric, _runs)
        pool.close()
        pool.join()
        docking_pathes = []
        for i in files:
            docking_path = os.path.join(os.path.dirname(i), 'pdbs_redock')
            if docking_path not in docking_pathes:
                docking_pathes.append(docking_path)
        score_files = []
        for docking_path in docking_pathes:
            score_files.append(os.path.join(docking_path,'interf_score.sc'))
        df =  statistic_redock(score_files, app=app)
    df.to_csv(out_path)
