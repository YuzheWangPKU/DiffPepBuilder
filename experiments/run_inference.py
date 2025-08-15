"""
Script to run de novo peptide generation.

Sample command:

> python experiments/inference.py

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
import tree
import numpy as np
import pandas as pd
import hydra
import torch

from torch.nn import DataParallel as DP
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from openfold.utils import rigid_utils
from hydra.core.hydra_config import HydraConfig

from analysis import utils as au
from data import utils as du
from data import pdb_data_loader
from data.pdb_data_loader import PdbDataset
from experiments.train import Experiment
from experiments.utils import save_traj
from omegaconf import DictConfig, OmegaConf
from SSbuilder.SSbuilder import Structure, read_ssdata, cys_terminal, build_ssbond, merge_s


class PepGenDataset(PdbDataset):
    def __init__(
            self,
            *,
            data_conf,
            sample_conf,
            diffuser
    ):
        super().__init__(data_conf=data_conf, diffuser=diffuser, is_training=False)
        self.sample_conf = sample_conf

        gen_csv = pd.DataFrame(np.repeat(self.csv.values, self._data_conf.num_repeat_per_eval_sample, axis=0))
        gen_csv.columns = self.csv.columns
        self.csv = gen_csv.copy()


    def sample_init_peptide(self, sample_length: int):
        """
        Sample initial peptide conformation based on length.

        Args:
            sample_length: length of the peptide to sample
        """
        res_mask = np.ones(sample_length)
        fixed_mask = np.zeros_like(res_mask)

        ref_sample = self.diffuser.sample_ref(
            n_samples=sample_length,
            as_tensor_7=True,
        )
        aatype = np.full(sample_length, 20)
        init_feats = {
            'aatype': aatype,
            'res_mask': res_mask,
            'fixed_mask': fixed_mask,
            'torsion_angles_sin_cos': np.zeros((sample_length, 7, 2)),
            'sc_ca_t': np.zeros((sample_length, 3)),
            **ref_sample,
        }
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats)
        
        return init_feats
    

    def process_receptor(self, raw_feats):
        """
        Process receptor features.
        """
        bb_rigid = rigid_utils.Rigid.from_tensor_4x4(
            raw_feats['rigidgroups_0'])[:, 0]
        fixed_mask = np.ones_like(raw_feats['ligand_mask'])
        rigids_0 = bb_rigid.to_tensor_7()
        sc_ca_t = torch.zeros_like(bb_rigid.get_trans())
        receptor_feats = {
            'aatype': raw_feats['aatype'],
            'res_mask': raw_feats['res_mask'],
            'fixed_mask': fixed_mask,
            'torsion_angles_sin_cos': raw_feats['torsion_angles_sin_cos'],
            'sc_ca_t': sc_ca_t,
            'rigids_t': rigids_0
        }
        receptor_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), receptor_feats)
        
        return receptor_feats
    

    def idx_to_peptide_len(self, idx):
        """
        Convert the index to the sampled peptide length.
        """
        min_length = self.sample_conf.min_length
        num_repeat = self.data_conf.num_repeat_per_eval_sample

        length = min_length + (idx % num_repeat)

        return length


    def __len__(self):
        return len(self.csv)


    def __getitem__(self, idx):
        csv_row = self.csv.iloc[idx]

        if 'pdb_name' in csv_row:
            pdb_name = csv_row['pdb_name'].split('_', 1)[0]
        else:
            raise ValueError('Need receptor identifier.')
        
        processed_file_path = csv_row['processed_path']

        raw_feats = self._process_csv_row(processed_file_path)

        peptide_len = self.idx_to_peptide_len(idx)

        peptide_feats = self.sample_init_peptide(peptide_len)
        receptor_feats = self.process_receptor(raw_feats)

        final_feats = tree.map_structure(
            lambda peptide_feat, receptor_feat: torch.cat((peptide_feat, receptor_feat), dim=0),
            peptide_feats,
            receptor_feats
        )

        # Add coordinate bias between the raw PDB file and the generated structure
        coordinate_bias = raw_feats['coordinate_bias']
        new_coordinate_bias = np.concatenate([coordinate_bias, np.tile(coordinate_bias[-1], (peptide_len, 1))], axis=0)
        final_feats['coordinate_bias'] = torch.tensor(new_coordinate_bias)

        # Generate chain indice and sequence indices for the complex
        receptor_chain_idx = raw_feats['chain_idx']
        receptor_seq_idx = raw_feats['seq_idx']

        peptide_chain_idx = np.zeros(peptide_len, dtype=int)
        peptide_seq_idx = np.arange(1, peptide_len + 1)

        # Update receptor chain indices and add a 100-residue gap between chains
        unique_chain_idx = np.unique(receptor_chain_idx)
        chain_idx_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_chain_idx, start=1)}
        new_receptor_chain_idx = np.vectorize(chain_idx_mapping.get)(receptor_chain_idx)
        new_receptor_seq_idx = receptor_seq_idx + peptide_len + 100

        chain_idx = np.concatenate([peptide_chain_idx, new_receptor_chain_idx])
        seq_idx = np.concatenate([peptide_seq_idx, new_receptor_seq_idx])

        final_feats['chain_idx'] = torch.tensor(chain_idx)
        final_feats['seq_idx'] = torch.tensor(seq_idx)
        
        # Use average ESM embeddings for peptide binder residues
        receptor_esm_embed = raw_feats['esm_embed']
        mean_esm_embed = np.mean(receptor_esm_embed[raw_feats['res_mask'].astype(bool)], axis=0)
        peptide_esm_embed = np.tile(mean_esm_embed, (peptide_len, 1))
        esm_embed = np.concatenate([peptide_esm_embed, receptor_esm_embed])
        final_feats['esm_embed'] = torch.tensor(esm_embed)

        final_feats = du.pad_feats(final_feats, csv_row['modeled_seq_len'] + peptide_len)

        return final_feats, pdb_name


class Sampler(Experiment):

    def __init__(
            self,
            conf: DictConfig,
        ):
        super().__init__(conf=conf)

        # Remove static type checking.
        OmegaConf.set_struct(conf, False)

        # Additional configs.
        self._infer_conf = conf.inference
        self._denoise_conf = self._infer_conf.denoising
        self._sample_conf = self._infer_conf.sampling
        self._ss_bond_conf = self._infer_conf.ss_bond

        # Set seed.
        seed = self._infer_conf.seed + self.ddp_info['local_rank'] if self._use_ddp else self._infer_conf.seed
        self._rng = np.random.default_rng(seed)

        # Reset number of repeats per receptor.
        num_repeat = self._sample_conf.max_length - self._sample_conf.min_length + 1
        self._conf.data.num_repeat_per_eval_sample = num_repeat
        self._data_conf.num_repeat_per_eval_sample = num_repeat
    

    def build_ss_bond(self, entropy_dict: dict) -> list:
        """
        Build disulfide bonds for generated peptide.
        """
        ligand_chain = ['A']
        receptor_chains = 'BCDEFGHIJKLMNOPQRSTUVWXYZ'
        pdb_ss_list = []

        if entropy_dict != {}:
            ss_frag_lib = read_ssdata(self._ss_bond_conf.frag_lib_path)
            for pdb_path in entropy_dict:
                pdb_ss_path = pdb_path.replace('.pdb', '_ss.pdb')
                peptide = Structure(pdb_path, chains = ligand_chain)
                receptor = Structure(pdb_path, chains = receptor_chains)
                if cys_terminal(peptide, entropy_dict[pdb_path], self._ss_bond_conf.entropy_threshold):
                    self._log.info(f"Try building SS bonds for {pdb_path}...")
                    new_peptide = None
                    try:
                        new_peptide = build_ssbond(peptide, ss_frag_lib, self._ss_bond_conf.max_ss_bond, 'A')
                    except Exception as e:
                        print(f"{pdb_path}")
                    if new_peptide is not None:
                        merge_s(receptor, new_peptide, pdb_ss_path)
                        pdb_ss_list.append(pdb_ss_path)
                        self._log.info(f"Successfully built SS bonds for {pdb_path}")
                    else:
                        self._log.info(f"Failed to build SS bond for {pdb_path}")

        return pdb_ss_list
    

    def run_sampling(self):
        """
        Set up inference run.
        """
        if HydraConfig.initialized() and 'num' in HydraConfig.get().job:
            replica_id = int(HydraConfig.get().job.num)
        else:
            replica_id = 0
        if self._use_wandb and replica_id == 0:
                self.init_wandb()
        assert(not self._exp_conf.use_ddp or self._exp_conf.use_gpu)

        # GPU mode
        if torch.cuda.is_available() and self._exp_conf.use_gpu:
            # Single GPU mode
            if self._exp_conf.num_gpus == 1:
                gpu_id = self._available_gpus[replica_id]
                device = f"cuda:{gpu_id}"
                self._model = self.model.to(device)
                self._log.info(f"Using device: {device}")
            # Multi gpu mode
            elif self._exp_conf.num_gpus > 1:
                device_ids = [
                f"cuda:{i}" for i in self._available_gpus[:self._exp_conf.num_gpus]
                ]
                # DDP mode
                if self._use_ddp :
                    device = torch.device("cuda",self.ddp_info['local_rank'])
                    self._model = self.model.to(device)
                    self._log.info(f"Multi-GPU sampling on GPUs in DDP mode, node_id : {self.ddp_info['node_id']}, devices: {device_ids}")
                # DP mode
                else:
                    if len(self._available_gpus) < self._exp_conf.num_gpus:
                        raise ValueError(f"require {self._exp_conf.num_gpus} GPUs, but only {len(self._available_gpus)} GPUs available ")
                    self._log.info(f"Multi-GPU sampling on GPUs in DP mode: {device_ids}")
                    gpu_id = self._available_gpus[replica_id]
                    device = f"cuda:{gpu_id}"
                    self._model = DP(self._model, device_ids=device_ids)
                    self._model = self.model.to(device)
        else:
            device = 'cpu'
            self._model = self.model.to(device)
            self._log.info(f"Using device: {device}")

        assert self._exp_conf.eval_ckpt_path is not None, "Need to specify inference checkpoint path."
        self._model.eval()

        test_dataset = PepGenDataset(
            data_conf=self._data_conf,
            sample_conf=self._sample_conf,
            diffuser=self._diffuser,
        )
        if not self._use_ddp:
            test_sampler = pdb_data_loader.TrainSampler(
                data_conf=self._data_conf,
                dataset=test_dataset,
                batch_size=self._exp_conf.eval_batch_size,
                sample_mode=self._exp_conf.sample_mode
            )
        else:
            test_sampler = pdb_data_loader.DistributedTrainSampler(
                data_conf=self._data_conf,
                dataset=test_dataset,
                batch_size=self._exp_conf.eval_batch_size,
                shuffle=False
            )
        num_workers = self._exp_conf.num_loader_workers
        test_loader = du.create_data_loader(
            test_dataset,
            sampler=test_sampler,
            np_collate=False,
            length_batch=False,
            batch_size=self._exp_conf.eval_batch_size if not self._use_ddp else self._exp_conf.eval_batch_size // self.ddp_info['world_size'],
            shuffle=False,
            num_workers=num_workers,
            drop_last=False
        )

        start_time = time.time()
        test_dir = self._exp_conf.eval_dir
        entropy_dict = {}

        for test_feats, pdb_names in test_loader:
            res_mask = du.move_to_np(test_feats['res_mask'].bool())
            fixed_mask = du.move_to_np(test_feats['fixed_mask'].bool())
            gt_aatype = du.move_to_np(test_feats['aatype'])
            seq_idx = du.move_to_np(test_feats['seq_idx'])
            chain_idx = du.move_to_np(test_feats['chain_idx'])
            coordinate_bias = du.move_to_np(test_feats['coordinate_bias'])
            batch_size = res_mask.shape[0]
            peptide_len = np.sum(1 - fixed_mask[0])
            test_feats = tree.map_structure(
                lambda x: x.to(device), test_feats)
            
            infer_out = self.inference_fn(
                data_init=test_feats,
                num_t=self._denoise_conf.num_t,
                min_t=self._denoise_conf.min_t,
                aux_traj=False,
                noise_scale=self._denoise_conf.noise_scale
            )

            final_prot = infer_out['prot_traj'][0]  # [N_res, 37, 3]
            temperature = max(self._sample_conf.seq_temperature, 1e-6)
            final_aa_prob = F.softmax(infer_out['aa_logits_pred'] / temperature, dim=-1)
            final_aatype = torch.multinomial(final_aa_prob.view(-1, self._model_conf.embed.num_aatypes), 1)
            final_aatype = du.move_to_np(final_aatype.view(batch_size, -1))

            if self._ss_bond_conf.save_entropy:
                final_aa_prob = torch.clamp(final_aa_prob, min=1e-10)
                final_aa_entropy = -torch.sum(final_aa_prob * torch.log(final_aa_prob), dim=-1)  # [batch_size, N_res]
                final_aa_entropy = du.move_to_np(final_aa_entropy)

            for i in range(batch_size):
                pdb_name = pdb_names[i]
                unpad_seq_idx = seq_idx[i][res_mask[i]]
                unpad_chain_idx = chain_idx[i][res_mask[i]]
                unpad_fixed_mask = fixed_mask[i][res_mask[i]]
                unpad_prot = final_prot[i][res_mask[i]]
                unpad_aatype = final_aatype[i][res_mask[i]]
                unpad_gt_aatype = gt_aatype[i][res_mask[i]]
                unpad_coordinate_bias = coordinate_bias[i][res_mask[i]]  # [N_res, 3]
                if self._ss_bond_conf.save_entropy:
                    unpad_aa_entropy = final_aa_entropy[i][res_mask[i]]

                length_dir = os.path.join(
                    test_dir,
                    f'{pdb_name}',
                    f'length_{peptide_len}'
                )
                os.makedirs(length_dir, exist_ok=True)
                sample_id = i + self.ddp_info['local_rank'] * batch_size if self._use_ddp else i
                pdb_sampled = os.path.join(
                    length_dir, 
                    f'{pdb_name}_sample_{sample_id}.pdb'
                )
                entropy_dict[pdb_sampled] = unpad_aa_entropy[~unpad_fixed_mask]
                b_factors = np.tile(unpad_aa_entropy[..., None], (1, 37)) * 30 if self._ss_bond_conf.save_entropy \
                    else np.tile(1 - unpad_fixed_mask[..., None], 37) * 100

                saved_path = au.write_prot_to_pdb(
                    unpad_prot,
                    pdb_sampled,
                    coordinate_bias=unpad_coordinate_bias,
                    aatype=np.where(unpad_fixed_mask == 0, unpad_aatype, unpad_gt_aatype),
                    residue_index=unpad_seq_idx,
                    chain_index=unpad_chain_idx,
                    no_indexing=True,
                    b_factors=b_factors
                )
                self._log.info(f'Done sample {pdb_name} (peptide length: {peptide_len}, sample: {sample_id}), saved to {saved_path}')

                if self._infer_conf.save_traj:
                    prot_traj = infer_out['prot_traj'][:, i, ...]  # [T, batch_size, N_res, 37, 3] -> [T, N_res, 37, 3]
                    unpad_prot_traj = prot_traj[:, res_mask[i], ...]
                    traj_path = save_traj(
                        bb_prot_traj=unpad_prot_traj,
                        coordinate_bias=unpad_coordinate_bias,  # [N_res, 3]
                        aatype=unpad_aatype,
                        diffuse_mask=1-unpad_fixed_mask,
                        prot_traj_path=os.path.join(length_dir, f'{pdb_name}_sample_{sample_id}_traj.pdb')
                    )
                    self._log.info(f'Saved denoising trajectory to {traj_path}')
        
        if self._use_ddp:
            dist.barrier()

        eval_time = time.time() - start_time
        pdb_list = list(entropy_dict.keys())
        self._log.info(f'Finished all de novo peptide generation tasks in {eval_time:.2f}s. Start post-processing...')

        # Build SS bond
        if self._ss_bond_conf.build_ss_bond:
            pdb_ss_list = self.build_ss_bond(entropy_dict)
            pdb_list += pdb_ss_list
            if self._use_ddp:
                dist.barrier()
            self._log.info(f'Finished building possible SS bonds for generated peptides.')


@hydra.main(version_base=None, config_path=f"{root}/config", config_name="inference")
def main(conf: DictConfig) -> None:
    sampler = Sampler(conf=conf)
    sampler.run_sampling()


if __name__ == '__main__':
    main()
