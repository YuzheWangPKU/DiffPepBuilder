"""
Pytorch script for running docking protocols.
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
import hydra
import torch
from collections import defaultdict

from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from openfold.utils import rigid_utils

from analysis import utils as au
from analysis.postprocess import Postprocess
from data import utils as du
from data import residue_constants
from data.pdb_data_loader import PdbDataset
from experiments.train import Experiment
from experiments.utils import save_traj


class BatchDockDataset(PdbDataset):
    def __init__(
            self,
            *,
            data_conf,
            diffuser
    ):
        super().__init__(data_conf=data_conf, diffuser=diffuser, is_training=False)


    def sample_init_peptide(self, peptide_seq: str):
        """
        Sample initial peptide conformation based on peptide sequence.
        """
        sample_length = len(peptide_seq)
        res_mask = np.ones(sample_length)
        fixed_mask = np.zeros_like(res_mask)

        ref_sample = self.diffuser.sample_ref(
            n_samples=sample_length,
            as_tensor_7=True,
        )
        aatype = np.array([residue_constants.restype_order.get(res, residue_constants.restype_num) for res in peptide_seq])
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
    

    def __len__(self):
        return len(self.csv)


    def __getitem__(self, idx):
        csv_row = self.csv.iloc[idx]

        if 'pdb_name' in csv_row:
            pdb_name = csv_row['pdb_name']
        else:
            raise ValueError('Need receptor identifier.')
        
        if 'peptide_id' in csv_row:
            peptide_id = csv_row['peptide_id']
        else:
            raise ValueError('Need peptide ligand identifier.')
        
        processed_file_path = csv_row['processed_path']
        raw_feats = self._process_csv_row(processed_file_path)

        peptide_seq = csv_row['peptide_seq']
        peptide_len = len(peptide_seq)
        peptide_feats = self.sample_init_peptide(peptide_seq)
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

        # ESM embeddings
        esm_embed = du.read_pkl(processed_file_path)['esm_embed']
        assert esm_embed.shape[0] == seq_idx.shape[0], \
            f"ESM embedding length {esm_embed.shape[0]} does not match sequence length {seq_idx.shape[0]}"
        final_feats['esm_embed'] = torch.tensor(esm_embed)

        final_feats = du.pad_feats(final_feats, csv_row['modeled_seq_len'])

        return final_feats, pdb_name, peptide_id


class EvalRepeatLengthSampler(torch.utils.data.Sampler):
    """
    Yields batches (lists of indices) such that:
      - each batch contains items of a single modeled_seq_len;
      - each target index appears num_repeat consecutive times;
      - DDP shards targets across ranks (disjoint ownership);
      - supports single-GPU and DP (loader_batch_size controls total per-process batch).
    """
    def __init__(self, *, dataset, num_repeat: int, batch_size: int,
                 world_size: int = 1, rank: int = 0, shuffle: bool = False):
        self.dataset = dataset
        self.csv = getattr(dataset, 'csv', None)
        assert self.csv is not None, "Dataset must expose a .csv DataFrame"
        self.num_repeat = int(num_repeat)
        self.batch_size = int(batch_size)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)

        # Build length bins (fallback to single bin if column absent)
        if 'modeled_seq_len' in self.csv.columns:
            by_len = {}
            for idx in range(len(self.csv)):
                L = int(self.csv.iloc[idx]['modeled_seq_len'])
                by_len.setdefault(L, []).append(idx)
        else:
            by_len = {0: list(range(len(self.csv)))}

        # Optional deterministic shuffle within each bin
        if self.shuffle:
            import random
            rng = random.Random(0)
            for L in by_len: rng.shuffle(by_len[L])

        # DDP rank sharding per bin
        for L, lst in by_len.items():
            by_len[L] = [lst[i] for i in range(self.rank, len(lst), self.world_size)]

        # Precompute batches: never cross bins
        batches = []
        for L in sorted(by_len.keys()):
            # Expand repeats, keeping repeats consecutive
            stream = []
            for idx in by_len[L]:
                stream.extend([idx] * self.num_repeat)

            # Cut into batches of size batch_size, last may be smaller
            p = 0
            N = len(stream)
            while p < N:
                q = min(p + self.batch_size, N)
                batches.append(stream[p:q])
                p = q

        self._batches = batches

    def __iter__(self):
        for b in self._batches:
            yield b

    def __len__(self):
        return len(self._batches)
    

class Sampler(Experiment):

    def __init__(
            self,
            conf: DictConfig,
        ):
        super().__init__(conf=conf)
        self._post_conf = conf.postprocess


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
        assert (not self._exp_conf.use_ddp or self._exp_conf.use_gpu)

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
                    model = self.model.to(device)
                    self._model = DDP(model, device_ids=[self.ddp_info['local_rank']], output_device=self.ddp_info['local_rank'], find_unused_parameters=True)
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

        test_dataset = BatchDockDataset(
            data_conf=self._data_conf,
            diffuser=self._diffuser
        )

        per_gpu_bs = self._exp_conf.eval_batch_size
        num_repeat = self._data_conf.get("num_repeat_per_eval_sample", 1)

        # Scenario (1): enforce clean per-GPU boundaries for repeats
        if num_repeat > per_gpu_bs:
            assert (num_repeat % per_gpu_bs) == 0, (
                f"When num_repeat_per_eval_sample ({num_repeat}) > per-GPU eval_batch_size ({per_gpu_bs}), "
                f"num_repeat_per_eval_sample must be divisible by eval_batch_size to avoid cross-target mixing within a GPU micro-batch."
            )

        # DDP topology (decouple ranks)
        if self._use_ddp:
            world_size = int(self.ddp_info['world_size'])
            rank = int(self.ddp_info['rank'])
        else:
            world_size, rank = 1, 0

        # Loader batch-size choice
        if torch.cuda.is_available() and self._exp_conf.use_gpu and self._exp_conf.num_gpus > 1 and not self._use_ddp:
            loader_batch_size = per_gpu_bs * self._exp_conf.num_gpus
        else:
            loader_batch_size = per_gpu_bs

        eval_batch_sampler = EvalRepeatLengthSampler(
            dataset=test_dataset,
            num_repeat=num_repeat,
            batch_size=loader_batch_size,
            world_size=world_size,
            rank=rank,
            shuffle=False
        )

        num_workers = self._exp_conf.num_loader_workers
        test_loader = du.create_data_loader(
            test_dataset,
            sampler=None,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            np_collate=False,
            length_batch=False,
            drop_last=False,
            batch_sampler=eval_batch_sampler
        )

        start_time = time.time()
        test_dir = self._exp_conf.eval_dir
        per_target_sample_ctr = defaultdict(int)

        for test_feats, pdb_names, peptide_ids in test_loader:
            res_mask = du.move_to_np(test_feats['res_mask'].bool())
            fixed_mask = du.move_to_np(test_feats['fixed_mask'].bool())
            gt_aatype = du.move_to_np(test_feats['aatype'])
            seq_idx = du.move_to_np(test_feats['seq_idx'])
            chain_idx = du.move_to_np(test_feats['chain_idx'])
            coordinate_bias = du.move_to_np(test_feats['coordinate_bias'])
            batch_size = res_mask.shape[0]
            test_feats = tree.map_structure(
                lambda x: x.to(device), test_feats)
            
            infer_out = self.inference_fn(
                data_init=test_feats,
                num_t=self._data_conf.num_t,
                min_t=self._data_conf.min_t,
                aux_traj=False,
                noise_scale=self._exp_conf.noise_scale
            )

            final_prot = infer_out['prot_traj'][0]  # [N_res, 37, 3]

            for i in range(batch_size):
                pdb_name = pdb_names[i]
                peptide_id = peptide_ids[i]
                unpad_seq_idx = seq_idx[i][res_mask[i]]
                unpad_chain_idx = chain_idx[i][res_mask[i]]
                unpad_fixed_mask = fixed_mask[i][res_mask[i]]
                unpad_prot = final_prot[i][res_mask[i]]
                unpad_gt_aatype = gt_aatype[i][res_mask[i]]
                unpad_coordinate_bias = coordinate_bias[i][res_mask[i]]  # [N_res, 3]

                peptide_seq_dir = os.path.join(
                    test_dir,
                    f'{pdb_name}',
                    f'{peptide_id}'
                )
                os.makedirs(peptide_seq_dir, exist_ok=True)

                sample_id = per_target_sample_ctr[(pdb_name, peptide_id)]
                per_target_sample_ctr[(pdb_name, peptide_id)] += 1

                pdb_sampled = os.path.join(
                    peptide_seq_dir, 
                    f'{pdb_name}_{peptide_id}_sample_{sample_id}.pdb'
                )
                b_factors = np.tile(1 - unpad_fixed_mask[..., None], 37) * 100

                saved_path = au.write_prot_to_pdb(
                    unpad_prot,
                    pdb_sampled,
                    coordinate_bias=unpad_coordinate_bias,
                    aatype=unpad_gt_aatype,
                    residue_index=unpad_seq_idx,
                    chain_index=unpad_chain_idx,
                    no_indexing=True,
                    b_factors=b_factors
                )
                self._log.info(f'Done sample {pdb_name} (peptide ligand id: {peptide_id}, sample: {sample_id}), saved to {saved_path}')

                # Postprocessing
                if self._post_conf.run_postprocess:
                    self._log.info(f'Postprocessing {pdb_name} (peptide ligand id: {peptide_id}, sample: {sample_id})...')
                    try:
                        postprocess = Postprocess(
                            saved_path,
                            "A",
                            ori_dir=self._post_conf.ori_pdbs,
                            out_dir=peptide_seq_dir,
                            xml=self._post_conf.xml_path,
                            amber_relax=self._post_conf.amber_relax,
                            rosetta_relax=self._post_conf.rosetta_relax
                        )
                        postprocess()
                        self._log.info(f'Postprocessing completed for {pdb_name} (peptide ligand id: {peptide_id}, sample: {sample_id})')
                    except Exception as e:
                        self._log.error(f'Postprocessing failed for {pdb_name} (peptide ligand id: {peptide_id}, sample: {sample_id}): {e}')
        
                # Save denoising trajectory
                if self._exp_conf.save_traj:
                    prot_traj = infer_out['prot_traj'][:, i, ...]  # [T, batch_size, N_res, 37, 3] -> [T, N_res, 37, 3]
                    unpad_prot_traj = prot_traj[:, res_mask[i], ...]
                    traj_path = save_traj(
                        bb_prot_traj=unpad_prot_traj,
                        coordinate_bias=unpad_coordinate_bias,
                        aatype=unpad_gt_aatype,
                        diffuse_mask=1-unpad_fixed_mask,
                        prot_traj_path=os.path.join(peptide_seq_dir, "traj", f'{pdb_name}_{peptide_id}_sample_{sample_id}_traj.pdb')
                    )
                    self._log.info(f'Saved denoising trajectory to {traj_path}')

        if self._use_ddp:
            dist.barrier()

        eval_time = time.time() - start_time
        self._log.info(f'Finished all peptide docking tasks in {eval_time:.2f}s.')


@hydra.main(version_base=None, config_path=f"{root}/config", config_name="docking")
def main(conf: DictConfig) -> None:
    exp = Sampler(conf=conf)
    exp.run_sampling()


if __name__ == '__main__':
    main()
