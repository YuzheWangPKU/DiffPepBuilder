"""
Pytorch script for model evaluation.
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
import time
import hydra

from omegaconf import DictConfig
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from hydra.core.hydra_config import HydraConfig

from experiments.train import Experiment


class Evaluation(Experiment):

    def __init__(
            self,
            conf: DictConfig,
        ):
        super().__init__(conf=conf)


    def start_evaluation(self):
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

        self._model.eval()
        valid_loader, valid_sampler = self.create_valid_dataset()

        start_time = time.time()
        eval_dir = self._exp_conf.eval_dir
        os.makedirs(eval_dir, exist_ok=True)
        metrics = self.eval_fn(
            eval_dir, valid_loader, device,
            noise_scale=self._exp_conf.noise_scale
        )
        all_metrics = self.aggregate_metrics(eval_dir)
        self._log.info(
            f"[Eval]: peptide_rmsd={all_metrics['peptide_rmsd'].mean()}, "
            f"sequence_recovery={all_metrics['sequence_recovery'].mean()}, "
            f"sequence_similarity={all_metrics['sequence_similarity'].mean()}"
        )
        eval_time = time.time() - start_time
        self._log.info(f'Finished evaluation in {eval_time:.2f}s')


@hydra.main(version_base=None, config_path=f"{root}/config", config_name="eval_dock")
def main(conf: DictConfig) -> None:
    exp = Evaluation(conf=conf)
    exp.start_evaluation()


if __name__ == '__main__':
    main()
