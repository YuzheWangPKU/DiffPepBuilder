"""
Pytorch script for model training and validation.
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
import torch
import GPUtil
import time
import tree
import numpy as np
import wandb
import copy
import hydra
import logging
import copy
import random
import pandas as pd

from collections import defaultdict
from collections import deque
from datetime import datetime
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.nn import DataParallel as DP
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from openfold.utils import rigid_utils as ru
from hydra.core.hydra_config import HydraConfig

from analysis import utils as au
from analysis import metrics
from data import pdb_data_loader
from data import se3_diffuser
from data import utils as du
from data import all_atom
from model import score_network
from experiments import utils as eu


class Experiment:

    def __init__(
            self,
            conf: DictConfig,
        ):
        """
        Initialize experiment.

        Args:
            conf: Experiment configuration.
        """
        self._log = logging.getLogger(__name__)
        self._available_gpus = ''.join(
            [str(x) for x in GPUtil.getAvailable(
                order='memory', limit = 8)])

        # Configs
        self._conf = conf
        self._exp_conf = conf.experiment
        if HydraConfig.initialized() and 'num' in HydraConfig.get().job:
            self._exp_conf.name = (
                f'{self._exp_conf.name}_{HydraConfig.get().job.num}')
        self._diff_conf = conf.diffuser
        self._model_conf = conf.model
        self._data_conf = conf.data
        self._use_wandb = self._exp_conf.use_wandb
        self._use_ddp = self._exp_conf.use_ddp

        # 1. initialize ddp info if in ddp mode
        # 2. silent rest of logger when use ddp mode
        # 3. silent wandb logger
        # 4. unset checkpoint path if rank is not 0 to avoid saving checkpoints
        if self._use_ddp:
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            self.ddp_info = eu.get_ddp_info()
            if self.ddp_info['rank'] not in [0, -1]:
                self._log.addHandler(logging.NullHandler())
                self._log.setLevel("ERROR")
                self._use_wandb = False
                self._exp_conf.ckpt_dir = None
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Warm starting
        ckpt_model = None
        ckpt_opt = None
        self.trained_epochs = 0
        self.trained_steps = 0

        if conf.experiment.warm_start:
            ckpt_path = conf.experiment.warm_start_ckpt_path
            self._log.info(f'Loading checkpoint from {ckpt_path}')
            ckpt_pkl = du.read_pkl(ckpt_path, use_torch=True, map_location=device)
            ckpt_model = ckpt_pkl['model']

            if conf.experiment.use_warm_start_conf:
                OmegaConf.set_struct(conf, False)
                conf = OmegaConf.merge(conf, ckpt_pkl['conf'])
                OmegaConf.set_struct(conf, True)

            # For compatibility with older checkpoints.
            if not conf.experiment.reset_training:
                if 'optimizer' in ckpt_pkl:
                    ckpt_opt = ckpt_pkl['optimizer']
                if 'epoch' in ckpt_pkl:
                    self.trained_epochs = ckpt_pkl['epoch']
                if 'step' in ckpt_pkl:
                    self.trained_steps = ckpt_pkl['step']

        self.start_step = self.trained_steps

        # Initialize experiment objects
        self._diffuser = se3_diffuser.SE3Diffuser(self._diff_conf)
        self._model = score_network.ScoreNetwork(self._model_conf, self.diffuser)

        if ckpt_model is not None:
            ckpt_model = {k.replace('module.', ''):v for k,v in ckpt_model.items()}
            self._model.load_state_dict(ckpt_model, strict=False)

        num_parameters = sum(p.numel() for p in self._model.parameters())
        self._log.info(f'Number of model parameters: {(num_parameters / 1e6):.2f} M')
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self._exp_conf.learning_rate)
        if ckpt_opt is not None:
            self._optimizer.load_state_dict(ckpt_opt)
            for state in self._optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

        dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm")

        if self._exp_conf.ckpt_dir is not None:
            # Set-up checkpoint location
            ckpt_dir = os.path.join(
                self._exp_conf.ckpt_dir,
                self._exp_conf.name,
                dt_string)
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir, exist_ok=True)
            self._exp_conf.ckpt_dir = ckpt_dir
            self._log.info(f'Checkpoints saved to: {ckpt_dir}')
        else:
            self._log.info('No checkpoints being saved.')

        if self._exp_conf.eval_dir is not None :
            eval_dir = os.path.join(
                self._exp_conf.eval_dir,
                self._exp_conf.name,
                dt_string)
            self._exp_conf.eval_dir = eval_dir
            self._log.info(f'Evaluation saved to: {eval_dir}')
        else:
            self._exp_conf.eval_dir = os.devnull
            self._log.info(f'Evaluation will not be saved.')
        self._aux_data_history = deque(maxlen=100)

    @property
    def diffuser(self):
        return self._diffuser

    @property
    def model(self):
        return self._model

    @property
    def conf(self):
        return self._conf
    

    def create_train_dataset(self):
        train_dataset = pdb_data_loader.PdbDataset(
            data_conf=self._data_conf,
            diffuser=self._diffuser,
            is_training=True
        )
        if not self._use_ddp:
            train_sampler = pdb_data_loader.TrainSampler(
                data_conf=self._data_conf,
                dataset=train_dataset,
                batch_size=self._exp_conf.batch_size,
                sample_mode=self._exp_conf.sample_mode,
            )
        else:
            train_sampler = pdb_data_loader.DistributedTrainSampler(
                data_conf=self._data_conf,
                dataset=train_dataset,
                batch_size=self._exp_conf.batch_size,
            )
        num_workers = self._exp_conf.num_loader_workers
        train_loader = du.create_data_loader(
            train_dataset,
            sampler=train_sampler,
            np_collate=False,
            length_batch=True,
            batch_size=self._exp_conf.batch_size if not self._exp_conf.use_ddp else self._exp_conf.batch_size // self.ddp_info['world_size'],
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            max_squared_res=self._exp_conf.max_squared_res,
        )
        return train_loader, train_sampler
    

    def create_valid_dataset(self):
        valid_dataset = pdb_data_loader.PdbDataset(
            data_conf=self._data_conf,
            diffuser=self._diffuser,
            is_training=False
        )
        if not self._use_ddp:
            valid_sampler = pdb_data_loader.TrainSampler(
                data_conf=self._data_conf,
                dataset=valid_dataset,
                batch_size=self._exp_conf.eval_batch_size,
                sample_mode=self._exp_conf.sample_mode
            )
        else:
            valid_sampler = pdb_data_loader.DistributedTrainSampler(
                data_conf=self._data_conf,
                dataset=valid_dataset,
                batch_size=self._exp_conf.eval_batch_size,
                shuffle=False
            )
        num_workers = self._exp_conf.num_loader_workers
        valid_loader = du.create_data_loader(
            valid_dataset,
            sampler=valid_sampler,
            np_collate=False,
            length_batch=False,
            batch_size=self._exp_conf.eval_batch_size if not self._exp_conf.use_ddp else self._exp_conf.eval_batch_size // self.ddp_info['world_size'],
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        return valid_loader, valid_sampler
    

    def init_wandb(self):
        self._log.info('Initializing Wandb.')
        conf_dict = OmegaConf.to_container(self._conf, resolve=True)
        wandb.init(
            project='se3-diffusion',
            name=self._exp_conf.name,
            config=dict(eu.flatten_dict(conf_dict)),
            dir=self._exp_conf.wandb_dir,
        )
        self._exp_conf.run_id = wandb.util.generate_id()
        self._exp_conf.wandb_dir = wandb.run.dir
        self._log.info(
            f'Wandb: run_id={self._exp_conf.run_id}, run_dir={self._exp_conf.wandb_dir}')


    def start_training(self, return_logs=False):
        # Set environment variables for which GPUs to use.
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
                    self._log.info(f"Multi-GPU training on GPUs in DDP mode, node_id : {self.ddp_info['node_id']}, devices: {device_ids}")
                # DP mode
                else:
                    if len(self._available_gpus) < self._exp_conf.num_gpus:
                        raise ValueError(f"require {self._exp_conf.num_gpus} GPUs, but only {len(self._available_gpus)} GPUs available ")
                    self._log.info(f"Multi-GPU training on GPUs in DP mode: {device_ids}")
                    gpu_id = self._available_gpus[replica_id]
                    device = f"cuda:{gpu_id}"
                    self._model = DP(self._model, device_ids=device_ids)
                    self._model = self.model.to(device)
        else:
            device = 'cpu'
            self._model = self.model.to(device)
            self._log.info(f"Using device: {device}")

        self._model.train()
        train_loader, train_sampler = self.create_train_dataset()
        valid_loader, valid_sampler = self.create_valid_dataset()

        logs = []
        for epoch in range(self.trained_epochs, self._exp_conf.num_epoch):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if valid_sampler is not None:
                valid_sampler.set_epoch(epoch)
            self.trained_epochs = epoch
            epoch_log = self.train_epoch(
                train_loader,
                valid_loader,
                device,
                return_logs=return_logs
            )
            if return_logs:
                logs.append(epoch_log)

        self._log.info('Done')
        return logs


    def update_fn(self, data):
        """Updates the state using some data and returns metrics."""
        self._optimizer.zero_grad()
        loss, aux_data = self.loss_fn(data)
        loss.backward()
        self._optimizer.step()
        return loss, aux_data


    def aggregate_metrics(self, eval_dir):
        if self._use_ddp:
            dist.barrier()
            if self.ddp_info['rank'] == 0:
                ckpt_metrics = pd.concat(
                    [pd.read_csv(os.path.join(eval_dir, f'metrics_{i}.csv')) for i in range(self.ddp_info['world_size'])]
                )
                ckpt_metrics = ckpt_metrics.sort_values(by=['gt_pdb', 'sample_id'])
                ckpt_metrics.to_csv(os.path.join(eval_dir, f'metrics.csv'), index=False)
                for i in range(self.ddp_info['world_size']):
                    os.remove(os.path.join(eval_dir, f'metrics_{i}.csv'))
            dist.barrier()
        all_metrics = pd.read_csv(os.path.join(eval_dir, 'metrics.csv'))
        return all_metrics


    def train_epoch(self, train_loader, valid_loader, device, return_logs=False):
        log_lossses = defaultdict(list)
        global_logs = []
        log_time = time.time()
        step_time = time.time()
        for train_feats in train_loader:
            train_feats = tree.map_structure(
                lambda x: x.to(device), train_feats)
            loss, aux_data = self.update_fn(train_feats)
            if return_logs:
                global_logs.append(loss)
            for k,v in aux_data.items():
                log_lossses[k].append(du.move_to_np(v))
            self.trained_steps += 1

            # Logging to terminal
            if self.trained_steps == 1 or self.trained_steps % self._exp_conf.log_freq == 0:
                elapsed_time = time.time() - log_time
                log_time = time.time()
                step_per_sec = self._exp_conf.log_freq / elapsed_time
                rolling_losses = tree.map_structure(np.mean, log_lossses)
                loss_log = ' '.join([
                    f'{k}={v[0]:.4f}'
                    for k,v in rolling_losses.items() if 'batch' not in k
                ])
                self._log.info(
                    f'[{self.trained_steps}]: {loss_log}, steps/sec={step_per_sec:.5f}')
                log_lossses = defaultdict(list)

            # Take checkpoint
            if (
                (self.trained_steps % self._exp_conf.ckpt_freq) == 0
                or (self._exp_conf.early_ckpt and self.trained_steps - self.start_step == 100)
            ):
                if self._exp_conf.ckpt_dir is not None:
                    ckpt_path = os.path.join(
                        self._exp_conf.ckpt_dir, f'step_{self.trained_steps}.pth')
                    du.write_checkpoint(
                        ckpt_path,
                        copy.deepcopy(self.model.state_dict()),
                        self._conf,
                        copy.deepcopy(self._optimizer.state_dict()),
                        self.trained_epochs,
                        self.trained_steps,
                        logger=self._log,
                        use_torch=True
                    )
                    self._log.info(f'Running evaluation of {ckpt_path}')

                # Run evaluation
                self._model.eval()
                start_time = time.time()
                eval_dir = os.path.join(
                    self._exp_conf.eval_dir, f'step_{self.trained_steps}')
                os.makedirs(eval_dir, exist_ok=True)
                ckpt_metrics = self.eval_fn(
                    eval_dir, valid_loader, device,
                    noise_scale=self._exp_conf.noise_scale
                )
                all_metrics = self.aggregate_metrics(eval_dir)
                self._log.info(
                    f"[Eval-{self.trained_steps}]: peptide_rmsd={all_metrics['peptide_rmsd'].mean()}, "
                    f"peptide_aligned_rmsd={all_metrics['peptide_aligned_rmsd'].mean()}, "
                    f"sequence_recovery={all_metrics['sequence_recovery'].mean()}, "
                    f"sequence_similarity={all_metrics['sequence_similarity'].mean()}"
                )
                eval_time = time.time() - start_time
                self._log.info(f'Finished evaluation in {eval_time:.2f}s')
                self._model.train()
            else:
                ckpt_metrics = None
                eval_time = None

            # Remote log to Wandb.
            if self._use_wandb:
                step_time = time.time() - step_time
                example_per_sec = self._exp_conf.batch_size / step_time
                step_time = time.time()
                wandb_logs = {
                    'loss': loss,
                    'aa_type_loss': aux_data['aa_type_loss'],
                    'rotation_loss': aux_data['rot_loss'],
                    'translation_loss': aux_data['trans_loss'],
                    'bb_atom_loss': aux_data['bb_atom_loss'],
                    'dist_mat_loss': aux_data['batch_dist_mat_loss'],
                    'ca_clash_loss': aux_data['ca_clash_loss'],
                    'chi_loss': aux_data['chi_loss'],
                    'batch_size': aux_data['examples_per_step'],
                    'res_length': aux_data['res_length'],
                    'examples_per_sec': example_per_sec,
                    'num_epochs': self.trained_epochs,
                }

                # Stratified losses
                wandb_logs.update(eu.t_stratified_loss(
                    du.move_to_np(train_feats['t']),
                    du.move_to_np(aux_data['batch_aa_type_loss']),
                    loss_name='aa_type_loss',
                ))

                wandb_logs.update(eu.t_stratified_loss(
                    du.move_to_np(train_feats['t']),
                    du.move_to_np(aux_data['batch_rot_loss']),
                    loss_name='rot_loss',
                ))

                wandb_logs.update(eu.t_stratified_loss(
                    du.move_to_np(train_feats['t']),
                    du.move_to_np(aux_data['batch_trans_loss']),
                    loss_name='trans_loss',
                ))

                wandb_logs.update(eu.t_stratified_loss(
                    du.move_to_np(train_feats['t']),
                    du.move_to_np(aux_data['batch_bb_atom_loss']),
                    loss_name='bb_atom_loss',
                ))

                wandb_logs.update(eu.t_stratified_loss(
                    du.move_to_np(train_feats['t']),
                    du.move_to_np(aux_data['batch_dist_mat_loss']),
                    loss_name='dist_mat_loss',
                ))

                wandb_logs.update(eu.t_stratified_loss(
                    du.move_to_np(train_feats['t']),
                    du.move_to_np(aux_data['batch_ca_clash_loss']),
                    loss_name='ca_clash_loss',
                ))

                wandb_logs.update(eu.t_stratified_loss(
                    du.move_to_np(train_feats['t']),
                    du.move_to_np(aux_data['batch_chi_loss']),
                    loss_name='chi_loss',
                ))

                if ckpt_metrics is not None:
                    wandb_logs['eval_time'] = eval_time
                    for metric_name in metrics.ALL_METRICS:
                        wandb_logs[metric_name] = ckpt_metrics[metric_name].mean()
                    eval_table = wandb.Table(
                        columns=ckpt_metrics.columns.to_list()+['structure'])
                    for _, row in ckpt_metrics.iterrows():
                        pdb_path = row['sample_path']
                        row_metrics = row.to_list() + [wandb.Molecule(pdb_path)]
                        eval_table.add_data(*row_metrics)
                    wandb_logs['sample_metrics'] = eval_table

                wandb.log(wandb_logs, step=self.trained_steps)

            if torch.isnan(loss):
                if self._use_wandb:
                    wandb.alert(
                        title="Encountered NaN loss",
                        text=f"Loss NaN after {self.trained_epochs} epochs, {self.trained_steps} steps"
                    )
                raise Exception(f'NaN encountered')

        if return_logs:
            return global_logs


    def eval_fn(self, eval_dir, valid_loader, device, min_t=None, num_t=None, noise_scale=1.0):
        ckpt_eval_metrics = []
        for valid_feats, pdb_names in valid_loader:
            res_mask = du.move_to_np(valid_feats['res_mask'].bool())
            fixed_mask = du.move_to_np(valid_feats['fixed_mask'].bool())
            seq_idx = du.move_to_np(valid_feats['seq_idx'])
            chain_idx = du.move_to_np(valid_feats['chain_idx'])
            gt_aatype = du.move_to_np(valid_feats['aatype'])
            gt_prot = du.move_to_np(valid_feats['atom37_pos'])
            coordinate_bias = du.move_to_np(valid_feats['coordinate_bias'])  # [batch_size, N_res, 3]
            batch_size = res_mask.shape[0]
            valid_feats = tree.map_structure(
                lambda x: x.to(device), valid_feats)

            # Run inference
            infer_out = self.inference_fn(
                valid_feats, min_t=min_t, num_t=num_t, noise_scale=noise_scale, aux_traj=True)
            
            final_prot = infer_out['prot_traj'][0]
            final_chi = du.move_to_np(infer_out['chi_pred'])

            # Extract argmax predicted aatype
            if self._data_conf.mask_lig_seq:
                temperature = max(self._exp_conf.seq_temperature, 1e-6)
                final_aa_prob = F.softmax(infer_out['aa_logits_pred'] / temperature, dim=-1)  # [batch_size, N_res, 20]
                final_aatype = torch.multinomial(final_aa_prob.view(-1, self._model_conf.embed.num_aatypes), 1)  # [batch_size * N_res, 1]
                final_aatype = du.move_to_np(final_aatype.view(batch_size, -1))  # [batch_size, N_res]
            else:
                final_aatype = gt_aatype

            for i in range(batch_size):
                pdb_name = pdb_names[i]
                num_res = int(np.sum(res_mask[i]).item())
                unpad_seq_idx = seq_idx[i][res_mask[i]]
                unpad_chain_idx = chain_idx[i][res_mask[i]]
                unpad_fixed_mask = fixed_mask[i][res_mask[i]]
                unpad_diffused_mask = 1 - unpad_fixed_mask
                unpad_prot = final_prot[i][res_mask[i]]
                unpad_gt_prot = gt_prot[i][res_mask[i]]
                unpad_aatype = final_aatype[i][res_mask[i]]
                unpad_gt_aatype = gt_aatype[i][res_mask[i]]
                unpad_coordinate_bias = coordinate_bias[i][res_mask[i]]  # [N_res, 3]

                sample_id = i + self.ddp_info['local_rank'] * batch_size if self._use_ddp else i

                saved_path = au.write_prot_to_pdb(
                    unpad_prot,
                    os.path.join(
                        eval_dir,
                        f'{pdb_name}_sample_{sample_id}.pdb'
                    ),
                    coordinate_bias=unpad_coordinate_bias,
                    aatype=np.where(unpad_fixed_mask == 0, unpad_aatype, unpad_gt_aatype),
                    residue_index=unpad_seq_idx,
                    chain_index=unpad_chain_idx,
                    no_indexing=True,
                    b_factors=np.tile(1 - unpad_fixed_mask[..., None], 37) * 100
                )
                try:
                    sample_metrics = metrics.protein_metrics(
                        pdb_path=saved_path,
                        atom37_pos=unpad_prot,
                        aatype=unpad_aatype,
                        gt_atom37_pos=unpad_gt_prot,
                        gt_aatype=unpad_gt_aatype,
                        diffuse_mask=unpad_diffused_mask,
                        flip_align=self._exp_conf.flip_align,
                    )
                except ValueError as e:
                    self._log.warning(
                        f'Failed evaluation of length {num_res} sample {i}: {e}')
                    sample_metrics = {}
                    continue
                
                sample_metrics['step'] = self.trained_steps
                sample_metrics['num_res'] = num_res
                sample_metrics['peptide_len'] = np.sum(unpad_diffused_mask)
                sample_metrics['gt_pdb'] = pdb_names[i]
                sample_metrics['sample_id'] = sample_id

                ckpt_eval_metrics.append(sample_metrics)

        # Save metrics as CSV.
        metrics_csv_name = f"metrics_{self.ddp_info['local_rank']}.csv" if self._use_ddp else "metrics.csv"
        eval_metrics_csv_path = os.path.join(eval_dir, metrics_csv_name)
        ckpt_eval_metrics = pd.DataFrame(ckpt_eval_metrics)
        ckpt_eval_metrics.to_csv(eval_metrics_csv_path, index=False)
        
        return ckpt_eval_metrics


    def _self_conditioning(self, batch):
        model_sc = self.model(batch)
        batch['sc_ca_t'] = model_sc['rigids'][..., 4:]
        return batch


    def loss_fn(self, batch):
        """
        Compute loss and auxiliary data.

        Args:
            batch: Batched data.
            model_out: Output of model run on batch.

        Returns:
            loss: Final training loss scalar.
            aux_data: Additional logging data.
        """
        if self._model_conf.embed.embed_self_conditioning and random.random() > 0.5:
            with torch.no_grad():
                batch = self._self_conditioning(batch)
        model_out = self.model(batch)
        bb_mask = batch['res_mask']  # [batch_size, N_res]
        fixed_mask = batch['fixed_mask']  # [batch_size, N_res]
        diffuse_mask = 1 - fixed_mask
        loss_mask = bb_mask * diffuse_mask
        batch_size, num_res = bb_mask.shape

        gt_rot_score = batch['rot_score']
        gt_trans_score = batch['trans_score']
        rot_score_scaling = batch['rot_score_scaling']
        trans_score_scaling = batch['trans_score_scaling']
        gt_aatype = batch['aatype']  # [batch_size, N_res]
        gt_atom37 = batch['atom37_pos']  # [batch_size, N_res, 37, 3]
        atom37_mask = batch['atom37_mask']  # [batch_size, N_res, 37]
        batch_loss_mask = torch.any(bb_mask, dim=-1)

        pred_rot_score = model_out['rot_score'] * diffuse_mask[..., None]
        pred_trans_score = model_out['trans_score'] * diffuse_mask[..., None]
        pred_aa_logits = model_out['aa_logits'] * diffuse_mask[..., None]  # [batch_size, N_res, 20]

        # Residue type prediction loss
        aa_type_ce = F.cross_entropy(pred_aa_logits.view(-1, 20), gt_aatype.view(-1), reduction='none')  # [batch_size * N_res]
        aa_type_ce = aa_type_ce.view(batch_size, num_res) * loss_mask  # [batch_size, N_res]
        aa_type_loss = torch.sum(
            aa_type_ce,
            dim=-1
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        aa_type_loss *= self._exp_conf.get('aa_type_loss_weight', 0)
        aa_type_loss *= batch['t'] < self._exp_conf.aa_type_loss_t_filter

        # Translation score loss
        trans_score_mse = (gt_trans_score - pred_trans_score)**2 * loss_mask[..., None]
        trans_score_loss = torch.sum(
            trans_score_mse / trans_score_scaling[:, None, None]**2,
            dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        # Translation x0 loss
        gt_trans_x0 = batch['rigids_0'][..., 4:] * self._exp_conf.coordinate_scaling
        pred_trans_x0 = model_out['rigids'][..., 4:] * self._exp_conf.coordinate_scaling
        trans_x0_loss = torch.sum(
            (gt_trans_x0 - pred_trans_x0)**2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        trans_loss = (
            trans_score_loss * (batch['t'] > self._exp_conf.trans_x0_threshold)
            + trans_x0_loss * (batch['t'] <= self._exp_conf.trans_x0_threshold)
        )
        trans_loss *= self._exp_conf.trans_loss_weight
        trans_loss *= int(self._diff_conf.diffuse_trans)

        # Rotation loss
        if self._exp_conf.separate_rot_loss:
            gt_rot_angle = torch.norm(gt_rot_score, dim=-1, keepdim=True)
            gt_rot_axis = gt_rot_score / (gt_rot_angle + 1e-6)

            pred_rot_angle = torch.norm(pred_rot_score, dim=-1, keepdim=True)
            pred_rot_axis = pred_rot_score / (pred_rot_angle + 1e-6)

            # Separate loss on the axis
            axis_loss = (gt_rot_axis - pred_rot_axis)**2 * loss_mask[..., None]
            axis_loss = torch.sum(
                axis_loss, dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + 1e-10)

            # Separate loss on the angle
            angle_loss = (gt_rot_angle - pred_rot_angle)**2 * loss_mask[..., None]
            angle_loss = torch.sum(
                angle_loss / rot_score_scaling[:, None, None]**2,
                dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + 1e-10)
            angle_loss *= self._exp_conf.rot_loss_weight
            angle_loss *= batch['t'] > self._exp_conf.rot_loss_t_threshold
            rot_loss = angle_loss + axis_loss
        else:
            rot_mse = (gt_rot_score - pred_rot_score)**2 * loss_mask[..., None]
            rot_loss = torch.sum(
                rot_mse / rot_score_scaling[:, None, None]**2,
                dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + 1e-10)
            rot_loss *= self._exp_conf.rot_loss_weight
            rot_loss *= batch['t'] > self._exp_conf.rot_loss_t_threshold
        rot_loss *= int(self._diff_conf.diffuse_rot)
                
        # Backbone atom loss
        pred_atom37_bb = model_out['atom37'][:, :, :5]
        gt_atom37_bb = gt_atom37[:, :, :5]
        atom37_bb_mask = atom37_mask[:, :, :5]

        bb_atom_loss_mask = atom37_bb_mask * loss_mask[..., None]
        bb_atom_loss = torch.sum(
            (pred_atom37_bb - gt_atom37_bb) ** 2 * bb_atom_loss_mask[..., None],
            dim=(-1, -2, -3)
        ) / (bb_atom_loss_mask.sum(dim=(-1, -2)) + 1e-10)
        bb_atom_loss *= self._exp_conf.bb_atom_loss_weight
        bb_atom_loss *= batch['t'] < self._exp_conf.bb_atom_loss_t_filter

        # Pairwise distance loss
        gt_flat_atoms = gt_atom37_bb.reshape([batch_size, num_res*5, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_atom37_bb.reshape([batch_size, num_res*5, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(bb_mask[:, :, None], (1, 1, 5))
        flat_loss_mask = flat_loss_mask.reshape([batch_size, num_res*5])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_loss_mask[:, None, :]

        flat_fixed_mask = torch.tile(fixed_mask[:, :, None], (1, 1, 5))
        flat_fixed_mask = flat_fixed_mask.reshape([batch_size, num_res*5])
        pair_fixed_mask = flat_fixed_mask[..., None] * flat_fixed_mask[:, None, :]
        pair_dist_mask = pair_dist_mask * (1 - pair_fixed_mask)

        # No pairwise distance loss beyond the distance threshold
        proximity_mask = gt_pair_dists < self._exp_conf.dist_threshold
        pair_dist_mask  = pair_dist_mask * proximity_mask

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,
            dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) - torch.sum(diffuse_mask, dim=-1) + 1e-10)
        dist_mat_loss *= self._exp_conf.dist_mat_loss_weight
        dist_mat_loss *= batch['t'] < self._exp_conf.dist_mat_loss_t_filter

        # CA clash loss
        pred_ca_dist = model_out['ca_dist']
        pred_ca_clash = pred_ca_dist < self._exp_conf.ca_clash_dist_threshold
        inter_ca_mask = diffuse_mask[:, None, :] * fixed_mask[:, :, None]
        pair_bb_mask = bb_mask[:, :, None] * bb_mask[:, None, :]
        ca_clash_mask = pred_ca_clash * pair_bb_mask * inter_ca_mask
        ca_clash_loss = torch.sum(
            1e3 * (pred_ca_dist - self._exp_conf.ca_clash_dist_threshold) ** 2 * ca_clash_mask,
            dim=(1, 2)
        ) / (torch.sum(pair_bb_mask * inter_ca_mask, dim=(1, 2)) + 1e-10)
        ca_clash_loss *= self._exp_conf.ca_clash_loss_weight
        ca_clash_loss *= batch['t'] < self._exp_conf.ca_clash_loss_t_filter

        # Chi angle prediction loss
        pred_chi = model_out['chi']  # [batch_size, N_res, 2, 2]
        gt_chi = batch['torsion_angles_sin_cos'][..., 3:5, :]  # [batch_size, N_res, 2, 2]
        gt_chi_alt = batch['alt_torsion_angles_sin_cos'][..., 3:5, :]

        diff_norm_gt = torch.norm(pred_chi - gt_chi, dim=-1)  # [batch_size, N_res, 2]
        diff_norm_alt_gt = torch.norm(pred_chi - gt_chi_alt, dim=-1)
        chi_mse = torch.minimum(diff_norm_gt ** 2, diff_norm_alt_gt ** 2) * loss_mask[..., None]
        chi_loss = torch.sum(
            5 * chi_mse,
            dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        # Angle norm loss, see AF2 Algorithm 27
        torsion_norm = model_out['torsion_norm']  # [batch_size, N_res, 5]
        anglenorm_mae = F.l1_loss(torsion_norm, torch.ones_like(torsion_norm), reduction='none') * loss_mask[..., None]
        anglenorm_loss = torch.sum(
            anglenorm_mae,
            dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        chi_loss += anglenorm_loss * self._exp_conf.anglenorm_weight
        chi_loss *= self._exp_conf.chi_loss_weight
        chi_loss *= batch['t'] < self._exp_conf.chi_loss_t_filter
             
        # Final loss
        final_loss = (
            aa_type_loss
            + rot_loss
            + trans_loss
            + bb_atom_loss
            + dist_mat_loss
            + ca_clash_loss
            + chi_loss
        )

        def normalize_loss(x):
            return x.sum() /  (batch_loss_mask.sum() + 1e-10)

        aux_data = {
            'batch_train_loss': final_loss,
            'batch_aa_type_loss': aa_type_loss,
            'batch_rot_loss': rot_loss,
            'batch_trans_loss': trans_loss,
            'batch_bb_atom_loss': bb_atom_loss,
            'batch_dist_mat_loss': dist_mat_loss,
            'batch_ca_clash_loss': ca_clash_loss,
            'batch_chi_loss': chi_loss,
            'total_loss': normalize_loss(final_loss),
            'aa_type_loss': normalize_loss(aa_type_loss),
            'rot_loss': normalize_loss(rot_loss),
            'trans_loss': normalize_loss(trans_loss),
            'bb_atom_loss': normalize_loss(bb_atom_loss),
            'dist_mat_loss': normalize_loss(dist_mat_loss),
            'ca_clash_loss': normalize_loss(ca_clash_loss),
            'chi_loss': normalize_loss(chi_loss),
            'examples_per_step': torch.tensor(batch_size),
            'res_length': torch.mean(torch.sum(bb_mask, dim=-1)),
        }

        # Maintain a history of the past N number of steps.
        # Helpful for debugging.
        self._aux_data_history.append({
            'aux_data': aux_data,
            'model_out': model_out,
            'batch': batch
        })

        assert final_loss.shape == (batch_size,)
        assert batch_loss_mask.shape == (batch_size,)
        return normalize_loss(final_loss), aux_data


    def _calc_trans_0(self, trans_score, trans_t, t):
        beta_t = self._diffuser._se3_diffuser._r3_diffuser.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        cond_var = 1 - torch.exp(-beta_t)
        return (trans_score * cond_var + trans_t) / torch.exp(-1/2*beta_t)


    def _set_t_feats(self, feats, t, t_placeholder):
        feats['t'] = t * t_placeholder
        rot_score_scaling, trans_score_scaling = self.diffuser.score_scaling(t)
        feats['rot_score_scaling'] = rot_score_scaling * t_placeholder
        feats['trans_score_scaling'] = trans_score_scaling * t_placeholder
        return feats


    def forward_traj(self, x_0, min_t, num_t):
        forward_steps = np.linspace(min_t, 1.0, num_t)[:-1]
        x_traj = [x_0]
        for t in forward_steps:
            x_t = self.diffuser.se3_diffuser._r3_diffuser.forward(
                x_traj[-1], t, num_t)
            x_traj.append(x_t)
        x_traj = torch.stack(x_traj, axis=0)
        return x_traj


    def inference_fn(
            self,
            data_init,
            num_t=None,
            min_t=None,
            center=False,
            aux_traj=False,
            self_condition=True,
            noise_scale=1.0,
        ):
        """
        Inference function.

        Args:
            data_init: Initial data values for sampling.
        """

        # Run reverse process.
        sample_feats = copy.deepcopy(data_init)
        device = sample_feats['rigids_t'].device
        if sample_feats['rigids_t'].ndim == 2:
            t_placeholder = torch.ones((1,)).to(device)
        else:
            t_placeholder = torch.ones(
                (sample_feats['rigids_t'].shape[0],)).to(device)
        if num_t is None:
            num_t = self._data_conf.num_t
        if min_t is None:
            min_t = self._data_conf.min_t
        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = 1 / num_t
        all_rigids = [du.move_to_np(copy.deepcopy(sample_feats['rigids_t']))]
        all_bb_prots = []
        all_trans_0_pred = []
        all_bb_0_pred = []
        with torch.no_grad():
            if self._model_conf.embed.embed_self_conditioning and self_condition:
                sample_feats = self._set_t_feats(
                    sample_feats, reverse_steps[0], t_placeholder)
                sample_feats = self._self_conditioning(sample_feats)
            for t in reverse_steps:
                if t > min_t:
                    sample_feats = self._set_t_feats(sample_feats, t, t_placeholder)
                    model_out = self.model(sample_feats)
                    rot_score = model_out['rot_score']
                    trans_score = model_out['trans_score']
                    rigid_pred = model_out['rigids']
                    if self._model_conf.embed.embed_self_conditioning:
                        sample_feats['sc_ca_t'] = rigid_pred[..., 4:]
                    fixed_mask = sample_feats['fixed_mask'] * sample_feats['res_mask']
                    diffuse_mask = (1 - sample_feats['fixed_mask']) * sample_feats['res_mask']
                    rigids_t = self.diffuser.reverse(
                        rigid_t=ru.Rigid.from_tensor_7(sample_feats['rigids_t']),
                        rot_score=du.move_to_np(rot_score),
                        trans_score=du.move_to_np(trans_score),
                        diffuse_mask=du.move_to_np(diffuse_mask),
                        t=t,
                        dt=dt,
                        center=center,
                        noise_scale=noise_scale,
                    )
                else:
                    model_out = self.model(sample_feats)
                    rigids_t = ru.Rigid.from_tensor_7(model_out['rigids'])
                sample_feats['rigids_t'] = rigids_t.to_tensor_7().to(device)
                if aux_traj:
                    all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))

                # Calculate x0 prediction derived from score predictions.
                gt_trans_0 = sample_feats['rigids_t'][..., 4:]
                pred_trans_0 = rigid_pred[..., 4:]
                trans_pred_0 = diffuse_mask[..., None] * pred_trans_0 + fixed_mask[..., None] * gt_trans_0
                psi_pred = model_out['psi']
                chi_pred = model_out['chi']
                aa_logits_pred = model_out['aa_logits']

                if aux_traj:
                    atom37_0 = all_atom.compute_backbone(
                        ru.Rigid.from_tensor_7(rigid_pred),
                        psi_pred
                    )[0]
                    all_bb_0_pred.append(du.move_to_np(atom37_0))
                    all_trans_0_pred.append(du.move_to_np(trans_pred_0))
                atom37_t = all_atom.compute_backbone(
                    rigids_t, psi_pred)[0]
                all_bb_prots.append(du.move_to_np(atom37_t))

        # Flip trajectory so that it starts from t=0.
        # This helps visualization.
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_bb_prots = flip(all_bb_prots)
        if aux_traj:
            all_rigids = flip(all_rigids)
            all_trans_0_pred = flip(all_trans_0_pred)
            all_bb_0_pred = flip(all_bb_0_pred)

        ret = {
            'prot_traj': all_bb_prots,
            'psi_pred': psi_pred,
            'chi_pred': chi_pred,
            'aa_logits_pred': aa_logits_pred
        }
        if aux_traj:
            ret['rigid_traj'] = all_rigids
            ret['trans_traj'] = all_trans_0_pred
            ret['rigid_0_traj'] = all_bb_0_pred
            
        return ret


@hydra.main(version_base=None, config_path=f"{root}/config", config_name="base")
def main(conf: DictConfig) -> None:
    # Fixes bug in https://github.com/wandb/wandb/issues/1525
    os.environ["WANDB_START_METHOD"] = "thread"

    exp = Experiment(conf=conf)
    exp.start_training()


if __name__ == '__main__':
    main()
