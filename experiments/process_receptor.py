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
import dataclasses
import pandas as pd
from Bio import PDB
from Bio.PDB import Select
import numpy as np
import mdtraj as md
import json
import warnings
from tqdm import tqdm

from data import utils as du
from data import errors
from data import parsers
from data import residue_constants
from experiments.process_dataset import PretrainedSequenceEmbedder


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
        "--peptide_info_path",
        type=str,
        default=None,
        help="Path to the JSON file containing peptide info."
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


def read_peptide_info(json_path, receptor_name):
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    receptor_name_lower = receptor_name.lower()
    if receptor_name_lower not in map(str.lower, data):
        return None, None, None
    
    receptor_data = next(value for key, value in data.items() if key.lower() == receptor_name_lower)

    motif = receptor_data.get('motif')
    hotspots = receptor_data.get('hotspots')
    lig_chain = receptor_data.get('lig_chain')

    return motif, hotspots, lig_chain


def renumber_rec_chain(pdb_path, json_path, in_place=False):
    """
    Re-number the receptor chain IDs for compatibility. Optionally save the modified files (PDB and JSON) in-place.
    Return True if the PDB and JSON files are updated.
    """
    parser = PDB.PDBParser()
    structure = parser.get_structure('', pdb_path)
    chain_ids = [chain.id for chain in structure[0]]
    
    pdb_name = os.path.basename(pdb_path).replace('.pdb', '').replace('_receptor', '').lower()
    out_pdb_path = pdb_path if in_place else pdb_path.replace('.pdb', '_modified.pdb')
    out_json_path = json_path if in_place else json_path.replace('.json', '_modified.json')
    chain_id_map = {}

    if 'A' in chain_ids:
        # PDB update. Shift characters by one position e.g., 'A'->'B', 'B'->'C', etc.
        new_chain_ids = []
        for i, chain in enumerate(structure[0]):
            chain.id = str(i)  # Temporary IDs to avoid conflicts
        for chain_id in chain_ids:
            if chain_id >= 'A':
                new_chain_id = chr((ord(chain_id) - ord('A') + 1) % 26 + ord('A'))
                new_chain_ids.append(new_chain_id)
                chain_id_map[chain_id] = new_chain_id
            else:
                new_chain_ids.append(chain_id)
                chain_id_map[chain_id] = chain_id
        for chain, new_id in zip(structure[0], new_chain_ids):
            chain.id = new_id
        io = PDB.PDBIO()
        io.set_structure(structure)
        io.save(out_pdb_path)

        # JSON update
        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)
        json_data_lower = {k.lower(): v for k, v in json_data.items()}
        if pdb_name in json_data_lower:
            receptor_data = json_data_lower[pdb_name]
            for key in ['motif', 'hotspots']:
                if key in receptor_data:
                    items = receptor_data[key].split('-')
                    updated_items = ['-'.join(chain_id_map.get(item[0], item[0]) + item[1:] for item in items)]
                    receptor_data[key] = ''.join(updated_items)
            with open(out_json_path, 'w') as updated_json_file:
                json.dump(json_data, updated_json_file, indent=4)

        return True
    
    else:
        return False


def resid_unique(res):
    if type(res) == str:
        res_ = res.split()
        return f'{res_[1]}_{res_[2]}'
    return f'{res.get_parent().id}{res.id[1]}'


class resSelector(Select):
    def __init__(self, res_ids):
        self.res_ids = res_ids
    def accept_residue(self, residue):
        resid = resid_unique(residue)
        if resid not in self.res_ids:
            return False
        else:
            return True
        

def get_rec_seq(entity, lch=None):
    aatype_rec, chain_id_rec = [], []
    for res in entity.get_residues():
        res_name = residue_constants.substitute_non_standard_restype.get(res.resname, res.resname)
        try:
            float(res_name)
            raise errors.DataError(f"Residue name should not be a number: {res.resname}")
        except ValueError:
            pass
        res_shortname = residue_constants.restype_3to1.get(res_name, '<unk>')
        if res.parent.id != lch:
            aatype_rec.append(res_shortname)
            chain_id_rec.append(res.parent.id)
    mask_rec = np.zeros(len(chain_id_rec))
    assert len(aatype_rec) == len(mask_rec) == len(chain_id_rec), "Shape mismatch when processing raw data!"

    return aatype_rec, mask_rec, chain_id_rec


def get_motif_center_pos(infile:str, motif=None, hotspots=None, lig_chain_str=None, pocket_cutoff=10):
    p = PDB.PDBParser(QUIET=1)
    struct = p.get_structure('', infile)[0]
    out_motif_file = infile.replace('.pdb', '_processed.pdb')

    seq_rec, mask_rec, chain_id_rec = get_rec_seq(struct, lig_chain_str)
    rec_residues = [i for i in struct.get_residues() if i.parent.id != lig_chain_str]
    rec_chain, ref_coords_ca = [], []

    if lig_chain_str:
        lig_coords_ca = [i['CA'].coord for i in struct.get_residues() if i.parent.id == lig_chain_str]
        ref_coords_ca = []
        for i in lig_coords_ca:
            for k, j in enumerate(rec_residues):
                if np.linalg.norm(j['CA'].coord - i) <= 8:
                    ref_coords_ca.append(j['CA'].coord)
    elif hotspots:
        io = PDB.PDBIO()
        io.set_structure(struct)
        io.save(out_motif_file, select=resSelector(hotspots))
        ref_struct = p.get_structure('', out_motif_file)[0]
        ref_coords_ca = [i['CA'].coord for i in ref_struct.get_residues()]
    elif motif:
        io = PDB.PDBIO()
        io.set_structure(struct)
        io.save(out_motif_file, select=resSelector(motif))
        ref_struct = p.get_structure('', out_motif_file)[0]
        ref_coords_ca = [i['CA'].coord for i in ref_struct.get_residues()]
    
    if ref_coords_ca:
        for i in ref_coords_ca:
            for k, j in enumerate(rec_residues):
                if np.linalg.norm(j['CA'].coord - i) <= pocket_cutoff:
                    rec_chain.append(resid_unique(j))
                    mask_rec[k] = 1
    else:
        warnings.warn(f"No reference ligand chain, motif or hotspots found for {os.path.basename(infile)}. "
                      f"Use the whole receptor.")
        ref_coords_ca = [i['CA'].coord for i in struct.get_residues()]
        rec_chain = [resid_unique(i) for i in struct.get_residues()]
        mask_rec = np.ones(len(rec_chain))

    io = PDB.PDBIO()
    io.set_structure(struct)
    io.save(out_motif_file, select=resSelector(rec_chain))

    center_pos = np.sum(np.array(ref_coords_ca), axis=0) / len(ref_coords_ca)
    struct = p.get_structure('', out_motif_file)[0]

    raw_seq_data = {}
    for i, chain_id in enumerate(chain_id_rec):
        if chain_id not in raw_seq_data:
            raw_seq_data[chain_id] = {"seq": "", "mask": []}
        raw_seq_data[chain_id]["seq"] += seq_rec[i]
        raw_seq_data[chain_id]["mask"].append(mask_rec[i])

    return struct, center_pos, raw_seq_data


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

    if args.peptide_info_path is not None:
        motif_str, hotspots, lig_chain_str = read_peptide_info(args.peptide_info_path, pdb_name)
    else:
        motif_str, hotspots, lig_chain_str = None, None, None

    if lig_chain_str:
        if motif_str or hotspots:
            warnings.warn(f"Find both reference ligand chain and motif / hotspots for {pdb_name}. "
                        f"The reference ligand {lig_chain_str} will be used in priority.")
    else:
        if args.peptide_info_path is not None:
            if renumber_rec_chain(file_path, args.peptide_info_path, in_place=True):
                motif_str, hotspots, lig_chain_str = read_peptide_info(args.peptide_info_path, pdb_name)
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

    
