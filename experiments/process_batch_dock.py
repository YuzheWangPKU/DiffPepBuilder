"""
Script to process receptor PDB file for subsequent peptide batch docking.
NOTE: If more than one of the following is provided, the order of priority is: lig_chain -> hotspots -> motif.
    Please remove the native peptide ligand chain if hotspots or motif is specified.
    We preserve the "hotspots" format as legacy.
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
import time
import argparse
import dataclasses
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

from data import utils as du
from data import errors
from data import parsers
from data import residue_constants
from experiments.process_dataset import PretrainedSequenceEmbedder
from experiments.process_receptor import read_receptor_info, renumber_rec_chain, get_motif_center_pos


def create_parser():
    parser = argparse.ArgumentParser(
        description="Process receptor PDB files."
    )

    parser.add_argument(
        '--pdb_dir',
        help='Path to directory with receptor PDB files.',
        type=str
    )

    parser.add_argument(
        '--write_dir',
        help='Path to write results to.',
        type=str,
        default=os.path.join(os.getcwd(), 'processed_pdbs')
    )

    parser.add_argument(
        "--receptor_info_path",
        type=str,
        default=None,
        help="Path to the JSON file containing receptor info."
    )

    parser.add_argument(
        "--peptide_seq_path",
        type=str,
        default=None,
        help="Path to the FASTA file containing peptide sequences to be docked. " \
        "If not provided, the script will use the ligand chain from the PDB file."
    )

    parser.add_argument(
        '--max_batch_size',
        help='Maximum batch size for ESM embedding.',
        type=int,
        default=32
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

    return parser


def read_peptide_seq(peptide_seq_path: str):
    """
    Read peptide sequences from a FASTA file.    
    Returns a dictionary with peptide names as keys and sequences as values.
    """
    peptide_seq_dict = {}
    with open(peptide_seq_path, 'r') as f:
        for record in SeqIO.parse(f, 'fasta'):
            peptide_id = record.id
            peptide_seq = str(record.seq)
            if peptide_id in peptide_seq_dict:
                raise ValueError(f"Duplicate peptide ID found: {peptide_id}")
            peptide_seq_dict[peptide_id] = peptide_seq

    return peptide_seq_dict


def extract_lig_seq(pdb_file: str, chain_id: str) -> str:
    """
    Parse the PDB and return the one-letter sequence of chain `chain_id`.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('PDB', pdb_file)
    model = structure[0]
    if chain_id not in model:
        raise ValueError(f"Chain {chain_id!r} not found in {pdb_file}")
    seq = []
    for res in model[chain_id]:
        if is_aa(res, standard=True):
            seq.append(residue_constants.restype_3to1.get(res.resname, 'X'))

    return ''.join(seq)


def process_file(file_path:str, write_dir:str, peptide_dict:dict, hotspot_cutoff:float=8, pocket_cutoff:float=10, receptor_info_path:str=None):
    """
    Process protein file into usable, smaller pickles.
    """
    pdb_name = os.path.basename(file_path).replace('.pdb', '')

    if receptor_info_path is not None:
        motif_str, hotspots, lig_chain_str = read_receptor_info(receptor_info_path, pdb_name)
        if lig_chain_str != "A":
            renumber_rec_chain(file_path, receptor_info_path, in_place=True)
            motif_str, hotspots, lig_chain_str = read_receptor_info(receptor_info_path, pdb_name)
    else:
        renumber_rec_chain(file_path, in_place=True)
        motif_str, hotspots, lig_chain_str = None, None, None

    if lig_chain_str:
        lig_chain_seq = extract_lig_seq(file_path, lig_chain_str)
        if motif_str or hotspots:
            warnings.warn(f"Find both reference ligand chain and motif / hotspots for {pdb_name}. "
                        f"The reference ligand {lig_chain_str} will be used in priority.")
    else:
        if peptide_dict is None:
            raise errors.DataError(f"No ligand chain or peptide sequences provided for {pdb_name}. "
                                   f"Please provide at least one of them.")
        if hotspots:
            hotspots = hotspots.split('-')
            if motif_str:
                warnings.warn(f"Both motif and hotspots are found for {pdb_name}. "
                            f"The hotspots will be used in priority over the motif.")
        elif motif_str:
            motif_str = motif_str.split('-')

    try:
        structure, center_pos, receptor_raw_seq = get_motif_center_pos(file_path, motif_str, hotspots, lig_chain_str, hotspot_cutoff, pocket_cutoff)
    except Exception as e:
        print(f'Failed to parse {pdb_name} with error {e}')
        return None, None

    # Extract all chains
    struct_chains = {}
    for chain in structure.get_chains():
        struct_chains[chain.id.upper()] = chain

    # Extract features
    struct_feats = []
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
        chain_dict = du.parse_chain_feats(chain_dict, center_pos=center_pos)
        struct_feats.append(chain_dict)
        chain_mask = np.zeros(complex_length)
        chain_mask[res_count: res_count + len(chain_dict['aatype'])] = 1
        chain_masks[chain_id_str] = chain_mask
        res_count += len(chain_dict['aatype']) 
    receptor_feats = du.concat_np_features(struct_feats, False)
    receptor_feats['center_pos'] = center_pos
    
    # Process geometry features
    receptor_aatype = receptor_feats['aatype']
    modeled_idx = np.where(receptor_aatype != 20)[0]
    if np.sum(receptor_aatype != 20) == 0:
        raise errors.LengthError('No modeled residues')
    receptor_feats['modeled_idx'] = modeled_idx
    receptor_feats['ligand_mask'] = np.zeros(complex_length)

    # Pair with peptide sequences
    all_metadata, all_raw_seq_data = [], {}

    if peptide_dict is None:
        peptide_dict = {"nat": lig_chain_seq}

    for peptide_id, peptide_seq in peptide_dict.items():
        for res in peptide_seq:
            if res not in residue_constants.restypes:
                raise errors.DataError(f"Invalid residue {res} in peptide sequence {peptide_id}")

        entry_name = f'{pdb_name}_{peptide_id}'
        raw_seq_dict = {"A":{"seq": peptide_seq, "mask": [1] * len(peptide_seq)}, **receptor_raw_seq}
        all_raw_seq_data[entry_name] = raw_seq_dict

        processed_path = os.path.join(write_dir, f'{entry_name}.pkl')
        metadata = {
            'pdb_name': pdb_name,
            'num_chains': len(struct_chains) + 1,
            'seq_len': len(peptide_seq) + len(receptor_aatype),
            'modeled_seq_len':  len(peptide_seq) + len(modeled_idx),
            'peptide_id': peptide_id,
            'peptide_seq': peptide_seq,
            'processed_path': os.path.abspath(processed_path),
        }
        all_metadata.append(metadata)

        du.write_pkl(processed_path, receptor_feats)

    return all_metadata, all_raw_seq_data


def process_serially(all_paths, write_dir, peptide_dict, hotspot_cutoff=8, pocket_cutoff=10, receptor_info_path=None):
    final_metadata = []
    final_raw_data = {}

    for i, file_path in enumerate(all_paths):
        try:
            start_time = time.time()
            all_metadata, all_raw_seq_data = process_file(
                file_path,
                write_dir,
                peptide_dict,
                hotspot_cutoff=hotspot_cutoff,
                pocket_cutoff=pocket_cutoff,
                receptor_info_path=receptor_info_path
            )
            elapsed_time = time.time() - start_time
            print(f'Finished {file_path} in {elapsed_time:2.2f}s')
            final_metadata.extend(all_metadata)
            final_raw_data.update(all_raw_seq_data)
        except errors.DataError as e:
            print(f'Failed {file_path}: {e}')

    return final_metadata, final_raw_data


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
    metadata_file_name = 'metadata_test.csv'
    metadata_path = os.path.join(write_dir, metadata_file_name)
    print(f'Files will be written to {write_dir}')

    if args.peptide_seq_path is not None:
        peptide_dict = read_peptide_seq(args.peptide_seq_path)
    else:
        peptide_dict = None

    all_metadata, all_raw_data = process_serially(
        all_file_paths,
        write_dir,
        peptide_dict,
        hotspot_cutoff=args.hotspot_cutoff,
        pocket_cutoff=args.pocket_cutoff,
        receptor_info_path=args.receptor_info_path
    )

    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_path, index=False)
    num_succeeded = len(all_metadata)
    total_num_files = total_num_paths * len(peptide_dict) if peptide_dict else total_num_paths
    print(f'Finished processing {num_succeeded}/{total_num_files} files. Start ESM embedding...')

    # Embed sequences with pretrained ESM
    esm_embedder = PretrainedSequenceEmbedder(max_batch_size=args.max_batch_size)
    all_esm_embs = esm_embedder(all_raw_data)

    for pkl_file_path, complex_seq_len in tqdm(zip(metadata_df['processed_path'], metadata_df['seq_len'])):
        pkl_data = du.read_pkl(pkl_file_path)
        entry_name = os.path.basename(pkl_file_path).replace('.pkl', '')
        esm_embed = all_esm_embs[entry_name].detach().cpu().numpy()
        assert esm_embed.shape[0] == complex_seq_len, \
            f"ESM embedding length {esm_embed.shape[0]} does not match sequence length {complex_seq_len} for {entry_name}"
        pkl_data['esm_embed'] = esm_embed
        du.write_pkl(pkl_file_path, pkl_data)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
