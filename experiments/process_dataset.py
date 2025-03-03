"""
Script to preprocess PDB datasets.
NOTE: This script will overwrite the raw data.
"""
import pyrootutils

# See: https://github.com/ashleve/pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True
)

import os
import wget
import copy
import argparse
import dataclasses
import pandas as pd
import numpy as np
import mdtraj as md
import time
import multiprocessing as mp
import functools as fn
from tqdm import tqdm
from Bio import PDB
from Bio.PDB import Select
import esm

import torch
from torch import nn as nn
from torch.utils.data import DataLoader

from data import utils as du
from data import errors
from data import parsers
from data import residue_constants


def create_parser():
    parser = argparse.ArgumentParser(
        description='PDB processing script.'
    )

    parser.add_argument(
        '--pdb_dir',
        help='Path to directory with PDB files.',
        type=str
    )

    parser.add_argument(
        '--num_processes',
        help='Number of processes.',
        type=int,
        default=16
    )

    parser.add_argument(
        '--write_dir',
        help='Path to write results to.',
        type=str,
        default=os.path.join(os.getcwd(), 'processed_pdbs')
    )

    parser.add_argument(
        '--hotspot_cutoff',
        help='Cutoff for hotspot residues.',
        type=int,
        default=8
    )

    parser.add_argument(
        '--pocket_cutoff',
        help='Cutoff for pocket residues.',
        type=int,
        default=10
    )

    parser.add_argument(
        '--max_batch_size',
        help='Maximum batch size for ESM embedding.',
        type=int,
        default=32
    )

    parser.add_argument(
        '--verbose',
        help='Whether to log everything.',
        action='store_true'
    )

    return parser


class PretrainedSequenceEmbedder(nn.Module):
    """
    Pretrained protein sequence encoder from ESM (Frozen, default esm2_t33_650M_UR50D).
    Protein sequence representations are pre-calculated and saved.
    """
    huggingface_card = {
        "esm2_t48_15B_UR50D": "facebook/esm2_t48_15B_UR50D",
        "esm2_t36_3B_UR50D": "facebook/esm2_t36_3B_UR50D",
        "esm2_t33_650M_UR50D": "facebook/esm2_t33_650M_UR50D",
        "esm1b_t33_650M_UR50S": "facebook/esm1b_t33_650M_UR50S"
    }

    def __init__(
            self,
            model_name: str = "esm2_t33_650M_UR50D",
            checkpoint_dir: str = f"{root}/experiments/checkpoints",
            truncation_seq_length: int = 2046,
            max_batch_size: int = 16
    ):
        super(PretrainedSequenceEmbedder, self).__init__()
        self.truncation_seq_length = truncation_seq_length
        self.max_batch_size = max_batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.alphabet = self._build_from_local_checkpoint(model_name, checkpoint_dir)
        self.model.eval()


    def _build_from_local_checkpoint(self, model_name: str, checkpoint_dir: str):
        """
        Build model from local checkpoint.

        Enable CPU offloading with FSDP.
        Adapted from: https://github.com/facebookresearch/esm/blob/main/examples/esm2_infer_fairscale_fsdp_cpu_offloading.py
        """
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        if model_name not in self.huggingface_card:
            raise NotImplementedError(f"Model {model_name} not implemented")
        
        model_path = os.path.join(checkpoint_dir, f"{model_name}.pt")
        regression_path = os.path.join(checkpoint_dir, f"{model_name}-contact-regression.pt")

        if not os.path.isfile(model_path):
            print(f"Model file {model_path} not found. Downloading...")
            wget.download(f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt", out=model_path)

        if not os.path.isfile(regression_path):
            print(f"Model file {regression_path} not found. Downloading...")
            wget.download(f"https://dl.fbaipublicfiles.com/fair-esm/regression/{model_name}-contact-regression.pt", out=regression_path)
        
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_path)
        model.to(self.device)

        return model, alphabet


    def forward(self, raw_data: dict):
        """
        Compute embeddings of protein sequences.
        Modified from: https://github.com/facebookresearch/esm/blob/main/scripts/extract.py
        """
        labels, sequences = [], []
        raw_data = copy.deepcopy(raw_data)

        for pdb_name, pdb_data in raw_data.items():
            for chain_id, seq_data in pdb_data.items():
                labels.append((pdb_name, chain_id))
                sequences.append(seq_data["seq"])

        dataset = [(label, seq) for label, seq in zip(labels, sequences)]
        batch_converter = self.alphabet.get_batch_converter(self.truncation_seq_length)
        data_loader = DataLoader(
            dataset, batch_size=self.max_batch_size, collate_fn=batch_converter, shuffle=False
        )
        print(f"Read sequence data with {len(dataset)} sequences")

        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(tqdm(data_loader, desc="Processing protein sequence batches")):
                print(
                    f"Processing {batch_idx + 1} of {1 + len(dataset) // self.max_batch_size} batches ({toks.size(0)} sequences)"
                )
                toks = toks.to(self.device, non_blocking=True)
                repr_layers_num = self.model.num_layers

                results = self.model(toks, repr_layers=[repr_layers_num])
                representations = results["representations"][repr_layers_num].to("cpu")

                for i, (pdb_name, chain_id) in enumerate(labels):
                    mask = torch.tensor(raw_data[pdb_name][chain_id]["mask"]).to("cpu")
                    truncate_len = min(self.truncation_seq_length, mask.shape[0])
                    # Call clone on tensors to ensure tensors are not views into a larger representation
                    # See https://github.com/pytorch/pytorch/issues/1995
                    raw_data[pdb_name][chain_id]["esm_embed"] = representations[i, 1 : truncate_len + 1][mask[: truncate_len].bool()].clone()

            concat_embs = {}
            for pdb_name, chains in raw_data.items():
                concat_embs[pdb_name] = torch.cat([chain["esm_embed"] for chain in chains.values()], dim=0)

        return concat_embs
    

def resid_unique(res):
    if type(res) == str:
        res_ = res.split()
        return f'{res_[1]}_{res_[2]}'
    return f'{res.id[1]}_{res.get_parent().id}'


def get_seq(entity, lch):
    aatype_rec, aatype_lig = [], []
    chain_id_rec, chain_id_lig = [], []
    for res in entity.get_residues():
        res_name = residue_constants.substitute_non_standard_restype.get(res.resname, res.resname)
        try:
            float(res_name)
            raise errors.DataError(f"Residue name should not be a number: {res.resname}")
        except ValueError:
            pass
        res_shortname = residue_constants.restype_3to1.get(res_name, '<unk>')
        # restype_idx = residue_constants.restype_order.get(res_shortname, residue_constants.restype_num)
        if res.parent.id == lch:
            aatype_lig.append(res_shortname)
            chain_id_lig.append(lch)
        elif res.parent.id != lch:
            aatype_rec.append(res_shortname)
            chain_id_rec.append(res.parent.id)
    aatype_full = aatype_lig + aatype_rec
    chain_id_full = chain_id_lig + chain_id_rec
    mask_full = np.concatenate([np.ones(len(chain_id_lig)), np.zeros(len(chain_id_rec))])
    assert len(aatype_full) == len(mask_full) == len(chain_id_full), "Shape mismatch when processing raw data!"

    return aatype_full, mask_full, chain_id_full


class resSelector(Select):
    def __init__(self, res_ids):
        self.res_ids = res_ids
    def accept_residue(self, residue):
        resid = resid_unique(residue)
        if resid not in self.res_ids:
            return False
        else:
            return True
        

def get_motif_center_pos(infile:str, lig_chain:str, hotspot_cutoff=8, pocket_cutoff=10):
    p = PDB.PDBParser(QUIET=1)
    struct = p.get_structure('', infile)[0]
    out_motif_file = infile.replace('.pdb', '_processed.pdb')

    seq_full, mask_full, chain_id_full = get_seq(struct, lig_chain)

    ligand_coords_ca = [i['CA'].coord for i in struct.get_residues() if i.parent.id == lig_chain]
    assert len(ligand_coords_ca) != 0, f"Specified ligand chain not found for {os.path.basename(infile)}"
    rec_residues = [i for i in struct.get_residues() if i.parent.id != lig_chain]
    motif_lig_chain = [resid_unique(res) for res in struct.get_residues() if res.parent.id == lig_chain]

    hotspot_coords_ca = []
    for i in ligand_coords_ca:
        for k, j in enumerate(rec_residues):
            if np.linalg.norm(j['CA'].coord - i) <= hotspot_cutoff:
                hotspot_coords_ca.append(j['CA'].coord)

    for i in hotspot_coords_ca:
        for k, j in enumerate(rec_residues):
            if np.linalg.norm(j['CA'].coord - i) <= pocket_cutoff:
                motif_lig_chain.append(resid_unique(j))
                mask_full[k + len(ligand_coords_ca)] = 1

    io = PDB.PDBIO()
    io.set_structure(struct)
    io.save(out_motif_file, select=resSelector(motif_lig_chain))

    center_pos = np.sum(np.array(ligand_coords_ca), axis=0) / len(ligand_coords_ca)
    struct = p.get_structure('', out_motif_file)[0]

    raw_seq_data = {}
    for i, chain_id in enumerate(chain_id_full):
        if chain_id not in raw_seq_data:
            raw_seq_data[chain_id] = {"seq": "", "mask": []}
        raw_seq_data[chain_id]["seq"] += seq_full[i]
        raw_seq_data[chain_id]["mask"].append(mask_full[i])

    return struct, center_pos, raw_seq_data


def process_file(file_path:str, write_dir:str, lig_chain_str:str='A', hotspot_cutoff:int=8, pocket_cutoff:int=10):
    """
    Processes protein file into usable, smaller pickles.

    Args:
        file_path: Path to file to read.
        write_dir: Directory to write pickles to.
        motif_str: 'A1-A2-A3-... -B4-...'
        hotspots: 'A1-A2-A3-... -B4-...'
        lig_chain_str: 'A'

    Returns:
        Saves processed protein to pickle and returns metadata.

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propogated.
    """
    metadata = {}
    pdb_name = os.path.basename(file_path).replace('.pdb', '')
    metadata['pdb_name'] = pdb_name
    lig_chain_str = pdb_name.split('_')[-2][0]
    # lig_chain_str = 'A'
    lig_chain_str = lig_chain_str if lig_chain_str.isalpha() else 'A'
    processed_path = os.path.join(write_dir, f'{pdb_name}.pkl')
    metadata['processed_path'] = os.path.abspath(processed_path)

    try:
        structure, center_pos, raw_seq_data = get_motif_center_pos(file_path, lig_chain_str, hotspot_cutoff=hotspot_cutoff, pocket_cutoff=pocket_cutoff)
        raw_seq_dict = {pdb_name: raw_seq_data}
    except Exception as e:
        print(f'Failed to parse {pdb_name} with error {e}')
        return None, None

    # Extract all chains
    struct_chains = {}
    struct_chains[lig_chain_str.upper()] = structure[lig_chain_str]
    for chain in structure.get_chains():
        if chain.id != lig_chain_str:
            struct_chains[chain.id.upper()] = chain
        
    com_center = center_pos
    metadata['num_chains'] = len(struct_chains)

    # Extract features
    struct_feats = []
    all_seqs = set()
    complex_length = 0
    for chain_id, chain in struct_chains.items():
        complex_length += len([i for i in chain.get_residues()]) 
    chain_masks =  {}
    res_count =  0
    for chain_id_str, chain in struct_chains.items():
        # Convert chain id into int
        chain_id = du.chain_str_to_int(chain_id_str)
        chain_prot = parsers.process_chain(chain, chain_id)
        chain_dict = dataclasses.asdict(chain_prot)
        chain_dict = du.parse_chain_feats(chain_dict, center_pos=com_center)
        all_seqs.add(tuple(chain_dict['aatype']))
        struct_feats.append(chain_dict)
        chain_mask = np.zeros(complex_length)
        chain_mask[res_count: res_count + len(chain_dict['aatype'])] = 1
        chain_masks[chain_id_str] = chain_mask
        res_count += len(chain_dict['aatype']) 
    if len(all_seqs) == 1:
        metadata['quaternary_category'] = 'homomer'
    else:
        metadata['quaternary_category'] = 'heteromer'
    complex_feats = du.concat_np_features(struct_feats, False)
    complex_feats['center_pos'] = center_pos
    
    # Process geometry features
    complex_aatype = complex_feats['aatype']
    metadata['seq_len'] = len(complex_aatype)
    modeled_idx = np.where(complex_aatype != 20)[0]
    if np.sum(complex_aatype != 20) == 0:
        raise errors.LengthError('No modeled residues')
    metadata['modeled_seq_len'] = len(modeled_idx)
    complex_feats['modeled_idx'] = modeled_idx
    if lig_chain_str is not None:
        try:
            complex_feats['ligand_mask'] = chain_masks[lig_chain_str]
        except KeyError:
            print(file_path, 'failed')
            return None, None
    else:
        # complex_feats['ligand_mask'] = np.zeros(complex_length)
        raise errors.DataError("No ligand chain specified")
    
    try:
        # MDtraj
        traj = md.load(file_path)
        # SS calculation
        pdb_ss = md.compute_dssp(traj, simplified=True)
        # DG calculation
        pdb_dg = md.compute_rg(traj)
        # os.remove(file_path)
    except Exception as e:
        # os.remove(file_path)
        raise errors.DataError(f'Mdtraj failed with error {e}')

    chain_dict['ss'] = pdb_ss[0]
    metadata['coil_percent'] = np.sum(pdb_ss == 'C') / metadata['modeled_seq_len']
    metadata['helix_percent'] = np.sum(pdb_ss == 'H') / metadata['modeled_seq_len']
    metadata['strand_percent'] = np.sum(pdb_ss == 'E') / metadata['modeled_seq_len']

    # Radius of gyration
    metadata['radius_gyration'] = pdb_dg[0]
    
    # Write features to pickles.
    du.write_pkl(processed_path, complex_feats)

    # Return metadata
    return metadata, raw_seq_dict


def process_serially(all_paths, write_dir, hotspot_cutoff=8, pocket_cutoff=10):
    all_metadata = []
    all_raw_data = {}

    for i, file_path in enumerate(all_paths):
        try:
            start_time = time.time()
            metadata, raw_seq_data = process_file(
                file_path,
                write_dir,
                hotspot_cutoff=hotspot_cutoff,
                pocket_cutoff=pocket_cutoff
            )
            elapsed_time = time.time() - start_time
            print(f'Finished {file_path} in {elapsed_time:2.2f}s')
            all_metadata.append(metadata)
            all_raw_data.update(raw_seq_data)
        except errors.DataError as e:
            print(f'Failed {file_path}: {e}')

    return all_metadata, all_raw_data


def process_fn(
        file_path,
        verbose=None,
        write_dir=None,
        hotspot_cutoff=8,
        pocket_cutoff=10        
        ):
    try:
        start_time = time.time()
        metadata, raw_seq_data = process_file(
            file_path,
            write_dir,
            hotspot_cutoff=hotspot_cutoff,
            pocket_cutoff=pocket_cutoff
        )
        elapsed_time = time.time() - start_time
        if verbose:
            print(f'Finished {file_path} in {elapsed_time:2.2f}s')
        return metadata, raw_seq_data
    except errors.DataError as e:
        if verbose:
            print(f'Failed {file_path}: {e}')
        return None, None


def main(args):
    pdb_dir = args.pdb_dir
    for file_name in os.listdir(pdb_dir):
        if file_name.endswith('_processed.pdb'):
            file_path = os.path.join(pdb_dir, file_name)
            try:
                os.remove(file_path)
                print(f'Removed file: {file_path}')
            except OSError as e:
                print(f'Error while deleting file {file_path}: {e}')
                
    all_file_paths = [
        os.path.join(pdb_dir, x)
        for x in os.listdir(args.pdb_dir) if '.pdb' in x]
    total_num_paths = len(all_file_paths)
    write_dir = args.write_dir
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    metadata_file_name = 'metadata.csv'
    metadata_path = os.path.join(write_dir, metadata_file_name)
    print(f'Files will be written to {write_dir}')

    # Process each pdb file
    if args.num_processes == 1:
        all_metadata, all_raw_data = process_serially(
            all_file_paths,
            write_dir,
            hotspot_cutoff=args.hotspot_cutoff,
            pocket_cutoff=args.pocket_cutoff
        )
    else:
        _process_fn = fn.partial(
            process_fn,
            verbose=args.verbose,
            write_dir=write_dir,
            hotspot_cutoff=args.hotspot_cutoff,
            pocket_cutoff=args.pocket_cutoff
        )
        with mp.Pool(processes=args.num_processes) as pool:
            results = pool.map(_process_fn, all_file_paths)
        all_metadata, all_raw_data = [], {}
        for metadata, raw_seq_data in results:
            if metadata is not None:
                all_metadata.append(metadata)
            if raw_seq_data is not None:
                all_raw_data.update(raw_seq_data)

    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_path, index=False)
    succeeded = len(all_metadata)
    print(f'Finished processing {succeeded}/{total_num_paths} files. Start ESM embedding...')

    # Embed sequences with pretrained ESM
    esm_embedder = PretrainedSequenceEmbedder(max_batch_size=args.max_batch_size)
    all_esm_embs = esm_embedder(all_raw_data)

    for pkl_file_path in tqdm(metadata_df['processed_path']):
        pkl_data = du.read_pkl(pkl_file_path)
        pdb_name = os.path.basename(pkl_file_path).replace('.pkl', '')
        esm_embed = all_esm_embs[pdb_name].detach().cpu().numpy()
        assert esm_embed.shape[0] == pkl_data["aatype"].shape[0], f"Shape mismatch for {pdb_name}!"
        pkl_data['esm_embed'] = esm_embed
        du.write_pkl(pkl_file_path, pkl_data)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)

