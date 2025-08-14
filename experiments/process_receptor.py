"""
Script to process receptor PDB file for subsequent de novo peptide generation.
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
import pandas as pd
import numpy as np
from tqdm import tqdm

from data import utils as du
from data import errors
from experiments.preprocess_utils import PretrainedSequenceEmbedder, normalize_receptor_inputs, get_motif_center_pos, featurize_structure


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


def process_file(file_path: str, write_dir: str, hotspot_cutoff: float=8, pocket_cutoff: float=10, receptor_info_path: str=None):
    pdb_name = os.path.basename(file_path).replace('.pdb', '')
    lig_chain_str, motif_list, hotspots_list, _ = normalize_receptor_inputs(
        file_path, receptor_info_path=receptor_info_path, peptide_dict={"_dummy":"A"}
    )

    structure, center_pos, raw_seq_data = get_motif_center_pos(
        file_path, motif=motif_list, hotspots=hotspots_list, lig_chain_str=lig_chain_str,
        hotspot_cutoff=hotspot_cutoff, pocket_cutoff=pocket_cutoff
    )
    feats, chain_masks, struct_chains, complex_len = featurize_structure(structure, center_pos)
    feats['ligand_mask'] = np.zeros(complex_len)

    processed_path = os.path.join(write_dir, f'{pdb_name}.pkl')
    du.write_pkl(processed_path, feats)

    metadata = {
        'pdb_name': pdb_name, 'processed_path': os.path.abspath(processed_path),
        'num_chains': len(struct_chains), 'seq_len': len(feats['aatype']),
        'modeled_seq_len': len(feats['modeled_idx']),
    }
    return metadata, {pdb_name: raw_seq_data}


def process_serially(all_paths, write_dir, hotspot_cutoff=8, pocket_cutoff=10, receptor_info_path=None):
    all_metadata = []
    all_raw_data = {}

    for i, file_path in enumerate(all_paths):
        try:
            start_time = time.time()
            metadata, raw_seq_data = process_file(
                file_path,
                write_dir,
                hotspot_cutoff=hotspot_cutoff,
                pocket_cutoff=pocket_cutoff,
                receptor_info_path=receptor_info_path
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
        hotspot_cutoff=args.hotspot_cutoff,
        pocket_cutoff=args.pocket_cutoff,
        receptor_info_path=args.receptor_info_path
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
