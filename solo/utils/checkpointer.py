import numpy as np
import os, json, torch

from torch import Tensor
from copy import deepcopy
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from solo.utils.misc import omegaconf_select
from pytorch_lightning.callbacks import Checkpoint


class ModelCheckpoint(Checkpoint):
    FILE_EXTENSION = ".ckpt"

    def __init__(
        self, cfg, dirpath=None, monitor=None,
        frequency=50, save_last_ckpt=True, save_top_module=True
    ):

        super().__init__()
        self.cfg = cfg
        self.last_ckpt = None
        self.monitor = monitor
        self.dirpath = dirpath
        self.frequency = frequency
        self.save_last_ckpt = save_last_ckpt
        self.save_top_module = save_top_module

        self.top_value_file_dir = None
        self.top_value = -torch.tensor(np.Inf)

    @staticmethod
    def add_and_assert_specific_cfg(cfg: DictConfig) -> DictConfig:
        cfg.checkpoint = omegaconf_select(cfg, "checkpoint", default={})
        cfg.checkpoint.enabled = omegaconf_select(cfg, "checkpoint.enabled", default=False)
        cfg.checkpoint.dir = omegaconf_select(cfg, "checkpoint.dir", default="trained_models")
        cfg.checkpoint.frequency = omegaconf_select(cfg, "checkpoint.frequency", default=50)
        cfg.checkpoint.save_last_ckpt = omegaconf_select(cfg, "checkpoint.save_last_ckpt", default=True)
        cfg.checkpoint.save_top_module = omegaconf_select(cfg, "checkpoint.save_top_module", default=True)
        cfg.checkpoint.monitor= omegaconf_select(cfg, "checkpoint.monitor", default='valid_acc_avg')
        return cfg

    def save_args(self):
        args = OmegaConf.to_container(self.cfg)
        args["wandb_run_id"] = getattr(self, "wandb_run_id", None)
        json_path = os.path.join(self.dirpath, "args.json")
        json.dump(args, open(json_path, "w"), default=lambda o: "<not serializable>", indent=4)

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        version = str(trainer.logger.version)
        self.wandb_run_id = version
        self.dirpath = os.path.join(self.dirpath, version)
        os.makedirs(self.dirpath, exist_ok=True)

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.save_args()

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        epoch = trainer.current_epoch
        if self.save_top_module:
            monitor_candidates = self._monitor_candidates(trainer)
            current_value = monitor_candidates.get(self.monitor)
            if current_value>self.top_value:
                top_value_filename = 'BestTrainer-Epoch-{}-{}-{:.3f}'.format(epoch, self.monitor, current_value)
                top_value_file_dir = os.path.join(self.dirpath, top_value_filename + self.FILE_EXTENSION)
                trainer.save_checkpoint(top_value_file_dir)
                if self.top_value_file_dir and self.top_value_file_dir != top_value_file_dir:
                    os.remove(self.top_value_file_dir)
                self.top_value = current_value
                self.top_value_file_dir = top_value_file_dir

        if self.save_last_ckpt:
            file_dir_last = os.path.join(self.dirpath, 'last' + self.FILE_EXTENSION)
            trainer.save_checkpoint(file_dir_last)

            if self.last_ckpt and self.last_ckpt != file_dir_last:
                os.remove(self.last_ckpt)
            self.last_ckpt = file_dir_last

        if self.frequency:
            monitor_candidates = self._monitor_candidates(trainer)
            current_value = monitor_candidates.get(self.monitor)
            if trainer.current_epoch % self.frequency == 0 and not epoch==0:
                frequency_dir = os.path.join(self.dirpath, 'Z_frequency')
                # frequency_filename = 'FrequencyTrainer-Epoch-{}-{}-{:.3f}'.format(epoch, self.monitor, current_value)
                frequency_filename = 'FrequencyTrainer-Epoch-{}'.format(epoch)
                file_dir_frequency = os.path.join(frequency_dir, frequency_filename + self.FILE_EXTENSION)
                trainer.save_checkpoint(file_dir_frequency)

    def _monitor_candidates(self, trainer: "pl.Trainer"):
        monitor_candidates = deepcopy(trainer.callback_metrics)
        epoch = monitor_candidates.get("epoch")
        monitor_candidates["epoch"] = epoch.int() if isinstance(epoch, Tensor) else torch.tensor(trainer.current_epoch)
        return monitor_candidates