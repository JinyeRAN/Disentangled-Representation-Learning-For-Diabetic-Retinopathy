from typing import Any, Dict, List, Sequence, Tuple

import omegaconf
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.methods.base import BaseMethod
from solo.utils.test_utils import valid_core


@torch.no_grad()
def initialize_momentum_params(online_net: nn.Module, momentum_net: nn.Module):
    params_online = online_net.parameters()
    params_momentum = momentum_net.parameters()
    for po, pm in zip(params_online, params_momentum):
        pm.data.copy_(po.data)
        pm.requires_grad = False

class MomentumUpdater:
    def __init__(self, base_tau: float = 0.996, final_tau: float = 1.0):
        super().__init__()

        assert 0 <= base_tau <= 1
        assert 0 <= final_tau <= 1 and base_tau <= final_tau

        self.base_tau = base_tau
        self.cur_tau = base_tau
        self.final_tau = final_tau

    @torch.no_grad()
    def update(self, online_net: nn.Module, momentum_net: nn.Module):
        for op, mp in zip(online_net.parameters(), momentum_net.parameters()):
            mp.data = self.cur_tau * mp.data + (1 - self.cur_tau) * op.data

    def update_tau(self, cur_step: int, max_steps: int):
        self.cur_tau = (self.final_tau - (self.final_tau - self.base_tau) * (math.cos(math.pi * cur_step / max_steps) + 1) / 2)

def mocov2plus(query, key, queue, temperature=0.1):
    pos = torch.einsum("nc,nc->n", [query, key]).unsqueeze(-1)
    neg = torch.einsum("nc,ck->nk", [query, queue])
    logits = torch.cat([pos, neg], dim=1)
    logits /= temperature
    targets = torch.zeros(query.size(0), device=query.device, dtype=torch.long)
    return F.cross_entropy(logits, targets)


class MoCoV2Plus(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.base_model = self._BACKBONES[cfg.backbone.name]

        self.backbone: nn.Module = self.base_model(cfg.method)
        self.backbone.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(self.backbone.inplanes, cfg.method_kwargs.proj_hiddim),nn.ReLU(),
            nn.Linear(cfg.method_kwargs.proj_hiddim, cfg.method_kwargs.proj_outdim))

        self.momentum_backbone: nn.Module = self.base_model(cfg.method)
        self.momentum_backbone.fc = nn.Identity()
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.momentum_backbone.inplanes, cfg.method_kwargs.proj_hiddim), nn.ReLU(),
            nn.Linear(cfg.method_kwargs.proj_hiddim, cfg.method_kwargs.proj_outdim))

        initialize_momentum_params(self.backbone, self.momentum_backbone)
        initialize_momentum_params(self.projector, self.momentum_projector)
        self.momentum_updater = MomentumUpdater(cfg.momentum.base_tau, cfg.momentum.final_tau)

        # create the queue
        self.register_buffer("queue", torch.randn(2, cfg.method_kwargs.proj_outdim, cfg.method_kwargs.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        if cfg.method_kwargs.classifier.enable:
            self.classifier: nn.Module = nn.Linear(self.backbone.inplanes, cfg.method_kwargs.classifier.classes)
            self.valid_step_outputs, self.train_step_outputs = [], []

    @property
    def learnable_params(self) -> List[dict]:
        learnable_params = [
            {"name": "bkb", "params": self.backbone.parameters(), },
            {"name": "prj", "params": self.projector.parameters(), },
        ]
        if self.cfg.method_kwargs.classifier.enable:
            learnable_params.append(
            {"name": "cls", "params": self.classifier.parameters(), "lr": self.classifier_lr, "weight_decay": 0, },
            )
        return learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        momentum_pairs = [(self.backbone, self.momentum_backbone), (self.projector, self.momentum_projector)]
        return momentum_pairs

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        batch_size = keys.shape[1]
        ptr = int(self.queue_ptr)  # type: ignore
        assert self.cfg.method_kwargs.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        keys = keys.permute(0, 2, 1)
        self.queue[:, :, ptr : ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.cfg.method_kwargs.queue_size  # move pointer
        self.queue_ptr[0] = ptr  # type: ignore

    def forward(self, X: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        if not self.no_channel_last: X = X.to(memory_format=torch.channels_last)
        feats_fc = self.backbone(X)
        feats_fc = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(feats_fc), 1)
        z = F.normalize(self.projector(feats_fc), dim=-1)
        logits = self.classifier(feats_fc.detach())
        return {"logits": logits, "feats": feats_fc, "z": z, "targets":targets}

    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict:
        if not self.no_channel_last: X = X.to(memory_format=torch.channels_last)
        feats_fc = self.momentum_backbone(X)
        feats_fc = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(feats_fc), 1)
        z = F.normalize(self.momentum_projector(feats_fc), dim=-1)
        return {"momentum_feats": feats_fc, "momentum_z": z}

    def shared_forward(self, X: torch.Tensor, targets: torch.Tensor) -> Dict:
        out = {}
        out.update(self.forward(X, targets))
        out.update(self.momentum_forward(X))
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        X, targets, index = batch
        X = [X] if isinstance(X, torch.Tensor) else X

        out = [self.shared_forward(x_i, targets) for x_i in X]
        out = {k: torch.cat([out[k] for out in out], dim=0) for k in out[0].keys()}

        q1, q2 = torch.chunk(out["z"], chunks=2, dim=0)
        k1, k2 = torch.chunk(out["momentum_z"], chunks=2, dim=0)
        queue = self.queue.clone().detach()
        nce_loss = (mocov2plus(q1, k2, queue[1], self.cfg.method_kwargs.temperature) +
                    mocov2plus(q2, k1, queue[0], self.cfg.method_kwargs.temperature)) / 2
        cls_loss = F.cross_entropy(out["logits"], out["targets"], ignore_index=-1)

        self._dequeue_and_enqueue(torch.stack((k1, k2)))
        self.log_dict({"train_cls_loss": cls_loss,"train_nce_loss": nce_loss})
        return nce_loss + cls_loss

    def on_train_start(self):
        self.last_step = 0

    def on_train_batch_end(self, outputs: Dict[str, Any], batch: Sequence[Any], batch_idx: int):
        if self.trainer.global_step > self.last_step:
            # update momentum backbone and projector
            momentum_pairs = self.momentum_pairs
            for mp in momentum_pairs:
                self.momentum_updater.update(*mp)
            # log tau momentum
            self.log("tau", self.momentum_updater.cur_tau)
            # update tau
            self.momentum_updater.update_tau(
                cur_step=self.trainer.global_step,
                max_steps=self.trainer.estimated_stepping_batches,
            )
        self.last_step = self.trainer.global_step

    def on_train_epoch_end(self) -> None:
        if self.cfg.method_kwargs.classifier.enable:
            if not len(self.train_step_outputs)==0:
                logits  = torch.cat([i['logits']  for i in self.train_step_outputs], dim=0)
                targets = torch.cat([i['targets'] for i in self.train_step_outputs], dim=0)
                train_acc = valid_core(logits, targets)
                self.log_dict({'train_acc_avg':train_acc,})
                self.train_step_outputs.clear()

    def validation_step(self, batch, batch_id):
        X, targets, index = batch
        out = self.forward(X, targets)

        if self.cfg.method_kwargs.classifier.enable:
            metrics = {'logits':out['logits'].detach().cpu(), 'targets':out['targets'].detach().cpu(),}
            self.valid_step_outputs.append(metrics)

    def on_validation_epoch_end(self):
        if self.cfg.method_kwargs.classifier.enable:
            if not len(self.valid_step_outputs) == 0:
                logits  = torch.cat([i['logits'] for i in self.valid_step_outputs], dim=0)
                targets = torch.cat([i['targets'] for i in self.valid_step_outputs], dim=0)
                valid_acc = valid_core(logits, targets)
                self.log_dict({'valid_acc_avg': valid_acc})
                self.valid_step_outputs.clear()