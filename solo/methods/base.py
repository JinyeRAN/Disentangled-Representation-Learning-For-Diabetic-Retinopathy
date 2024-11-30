import torch, omegaconf
import pytorch_lightning as pl
from functools import partial
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

from solo.utils.misc import omegaconf_select
from solo.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from solo.backbones import (resnet18Z,resnet34Z,resnet50Z,swin_base,swin_large,swin_small,swin_tiny, resnet18,resnet34,resnet50,inception)


def static_lr(get_lr: Callable, param_group_indexes: Sequence[int], lrs_to_replace: Sequence[float],):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs


class BaseMethod(pl.LightningModule):
    _BACKBONES = {
        "resnet18": resnet18, "resnet50": resnet50, "resnet34": resnet34,"inception":inception,
        "resnet18Z": resnet18Z, "resnet50Z": resnet50Z, "resnet34Z": resnet34Z,
        "swin_tiny": swin_tiny,"swin_small": swin_small, "swin_base": swin_base, "swin_large": swin_large,
    }
    _OPTIMIZERS = {
        "sgd": torch.optim.SGD, "adam": torch.optim.Adam, "adamw": torch.optim.AdamW,
    }
    _SCHEDULERS = ["reduce", "warmup_cosine", "exponential", "none",]

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__()
        cfg = self.add_and_assert_specific_cfg(cfg)
        self.cfg: omegaconf.DictConfig = cfg

        ##############################
        # training related
        self.max_epochs: int = cfg.max_epochs
        self.accumulate_grad_batches: Union[int, None] = cfg.accumulate_grad_batches

        # optimizer related
        self.optimizer: str = cfg.optimizer.name
        self.batch_size: int = cfg.optimizer.batch_size
        self.lr: float = cfg.optimizer.lr
        self.weight_decay: float = cfg.optimizer.weight_decay
        self.classifier_lr: float = cfg.optimizer.classifier_lr
        self.extra_optimizer_args: Dict[str, Any] = cfg.optimizer.kwargs

        # scheduler related
        self.scheduler: str = cfg.scheduler.name
        self.lr_decay_steps: Union[List[int], None] = cfg.scheduler.lr_decay_steps
        self.min_lr: float = cfg.scheduler.min_lr
        self.warmup_start_lr: float = cfg.scheduler.warmup_start_lr
        self.warmup_epochs: int = cfg.scheduler.warmup_epochs
        self.scheduler_interval: str = cfg.scheduler.interval

        # if accumulating gradient then scale lr
        if self.accumulate_grad_batches:
            self.lr = self.lr * self.accumulate_grad_batches
            self.classifier_lr = self.classifier_lr * self.accumulate_grad_batches
            self.min_lr = self.min_lr * self.accumulate_grad_batches
            self.warmup_start_lr = self.warmup_start_lr * self.accumulate_grad_batches
        self.no_channel_last = cfg.performance.disable_channel_last

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        cfg.backbone.kwargs = omegaconf_select(cfg, "backbone.kwargs", {})

        # default parameters for optimizer
        cfg.optimizer.kwargs = omegaconf_select(cfg, "optimizer.kwargs", {})
        cfg.accumulate_grad_batches = omegaconf_select(cfg, "accumulate_grad_batches", 1)

        # default parameters for the scheduler
        cfg.scheduler.lr_decay_steps = omegaconf_select(cfg, "scheduler.lr_decay_steps", None)
        cfg.scheduler.min_lr = omegaconf_select(cfg, "scheduler.min_lr", 0.0)
        cfg.scheduler.warmup_start_lr = omegaconf_select(cfg, "scheduler.warmup_start_lr", 3e-5)
        cfg.scheduler.warmup_epochs = omegaconf_select(cfg, "scheduler.warmup_epochs", 10)
        cfg.scheduler.interval = omegaconf_select(cfg, "scheduler.interval", "step")

        # default parameters for performance optimization
        cfg.performance = omegaconf_select(cfg, "performance", {})
        cfg.performance.disable_channel_last = omegaconf_select(cfg, "performance.disable_channel_last", False)

        # default empty parameters for method-specific kwargs
        cfg.method_kwargs = omegaconf_select(cfg, "method_kwargs", {})

        return cfg

    def configure_optimizers(self) -> Tuple[List, List]:
        learnable_params = self.learnable_params
        idxs_no_scheduler = [i for i, m in enumerate(learnable_params) if m.pop("static_lr", False)]
        optimizer = self._OPTIMIZERS[self.optimizer]
        optimizer = optimizer(learnable_params, lr=self.lr, weight_decay=self.weight_decay,**self.extra_optimizer_args,)
        if self.scheduler.lower() == "none": return optimizer

        max_warmup_steps = (
            self.warmup_epochs * (self.trainer.estimated_stepping_batches / self.max_epochs)
            if self.scheduler_interval == "step" else self.warmup_epochs
        )
        max_scheduler_steps = (
            self.trainer.estimated_stepping_batches if self.scheduler_interval == "step" else self.max_epochs
        )
        scheduler = {
            "scheduler": LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=max_warmup_steps, max_epochs=max_scheduler_steps,
                warmup_start_lr=self.warmup_start_lr if self.warmup_epochs > 0 else self.lr, eta_min=self.min_lr,
            ),
            "interval": self.scheduler_interval,
            "frequency": 1,
        }

        if idxs_no_scheduler:
            partial_fn = partial(
                static_lr, get_lr=scheduler["scheduler"].get_lr if isinstance(scheduler, dict) else scheduler.get_lr,
                param_group_indexes=idxs_no_scheduler, lrs_to_replace=[self.lr] * len(idxs_no_scheduler),
            )
            if isinstance(scheduler, dict): scheduler["scheduler"].get_lr = partial_fn
            else: scheduler.get_lr = partial_fn

        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, *_):
        try:
            optimizer.zero_grad(set_to_none=True)
        except:
            optimizer.zero_grad()