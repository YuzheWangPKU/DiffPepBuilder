"""
Script to process receptor PDB file for subsequent peptide batch docking.
NOTE: If more than one of the following is provided, the order of priority is: lig_chain -> hotspots -> motif.
    Please remove the native peptide ligand chain if hotspots or motif is specified. 
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

from data import utils as du
from data import errors
from data import parsers
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
        help="Path to the FASTA file containing peptide sequences to be docked."
    )

    parser.add_argument(
        '--max_batch_size',
        help='Maximum batch size for ESM embedding.',
        type=int,
        default=32
    )
    
    parser.add_argument(
        '--pocket_cutoff',
        help='Cutoff for pocket residues.',
        type=int,
        default=10
    )

    return parser


def process_file(file_path:str, write_dir:str, pocket_cutoff:int=10):
    """
    Process protein file into usable, smaller pickles.

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
    pdb_name = os.path.basename(file_path).replace('.pdb', '').replace('_receptor', '')

    if args.receptor_info_path is not None:
        renumber_rec_chain(file_path, args.receptor_info_path, in_place=True)
        motif_str, hotspots, lig_chain_str = read_receptor_info(args.receptor_info_path, pdb_name)
    else:
        motif_str, hotspots, lig_chain_str = None, None, None

    if lig_chain_str:
        if motif_str or hotspots:
            warnings.warn(f"Find both reference ligand chain and motif / hotspots for {pdb_name}. "
                        f"The reference ligand {lig_chain_str} will be used in priority.")
    else:
        if hotspots:
            hotspots = hotspots.split('-')
            if motif_str:
                warnings.warn(f"Both motif and hotspots are found for {pdb_name}. "
                            f"The hotspots will be used in priority over the motif.")
        elif motif_str:
            motif_str = motif_str.split('-')

    metadata = {}
    metadata['pdb_name'] = pdb_name
    processed_path = os.path.join(write_dir, f'{pdb_name}.pkl')
    metadata['processed_path'] = os.path.abspath(processed_path)

    try:
        structure, center_pos, raw_seq_data = get_motif_center_pos(file_path, motif_str, hotspots, lig_chain_str, pocket_cutoff)
        raw_seq_dict = {pdb_name: raw_seq_data}
    except Exception as e:
        print(f'Failed to parse {pdb_name} with error {e}')
        return None, None

    # Extract all chains
    struct_chains = {}
    for chain in structure.get_chains():
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
    complex_feats['ligand_mask'] = np.zeros(complex_length)
    
    # Write features to pickles.
    du.write_pkl(processed_path, complex_feats)

    # Return metadata
    return metadata, raw_seq_dict


def process_serially(all_paths, write_dir, pocket_cutoff=10):
    all_metadata = []
    all_raw_data = {}

    for i, file_path in enumerate(all_paths):
        try:
            start_time = time.time()
            metadata, raw_seq_data = process_file(
                file_path,
                write_dir,
                pocket_cutoff=pocket_cutoff
            )
            elapsed_time = time.time() - start_time
            print(f'Finished {file_path} in {elapsed_time:2.2f}s')
            all_metadata.append(metadata)
            all_raw_data.update(raw_seq_data)
        except errors.DataError as e:
            print(f'Failed {file_path}: {e}')

    return all_metadata, all_raw_data


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

    all_metadata, all_raw_data = process_serially(
        all_file_paths,
        write_dir,
        pocket_cutoff=args.pocket_cutoff
    )

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
