from typing import Any, Dict, List, Sequence

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

from solo.methods.base import BaseMethod
from solo.methods.utils import BatchNorm1d
from solo.utils.test_utils import valid_core
from solo.losses.simclr import simclr_loss_func


class SupConBackbone(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.cfg = cfg

        ####################### backbone #########################
        self.base_model = self._BACKBONES[cfg.backbone.name]
        self.backbone = self.base_model(cfg.method)
        backbone_dim = self.backbone.inplanes
        self.backbone.fc = nn.Identity()

        # ####################### projector ########################
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, cfg.method_kwargs.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.method_kwargs.proj_hidden_dim, cfg.method_kwargs.proj_output_dim),
        )

        ###################### classifier ######################
        if cfg.method_kwargs.classifier.enable:
            self.classifier: nn.Module = nn.Linear(backbone_dim, cfg.method_kwargs.classifier.classes)
            self.valid_step_outputs = []
            self.train_step_outputs = []

    @property
    def learnable_params(self) -> List[dict]:
        learnable_params = [
            {"name": "bkb", "params": self.backbone.parameters(), },
            {"name": "prj", "params": self.projector.parameters(),},
        ]
        if self.cfg.method_kwargs.classifier.enable:
            learnable_params.append(
            {"name": "cls", "params": self.classifier.parameters(), "lr": self.classifier_lr, "weight_decay": 0, },
            )
        return learnable_params

    def forward(self, X, targets) -> Dict:
        if not self.no_channel_last: X = X.to(memory_format=torch.channels_last)
        feats_fc = self.backbone(X)
        feats_fc = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(feats_fc), 1)
        output = {"feats_fc": feats_fc, "targets": targets}
        if self.cfg.method_kwargs.classifier.enable:
            logits = self.classifier(feats_fc.detach())
            output.update({'logits': logits})

        return output

    def training_step(self, batch: Sequence[Any]):
        X, targets, index = batch
        X = [X] if isinstance(X, torch.Tensor) else X

        out = [self.forward(x_i, targets) for x_i in X]
        feats_fc = torch.cat([out_i["feats_fc"] for out_i in out], dim=0)
        targets = torch.cat([out_i["targets"] for out_i in out], dim=0)

        z = self.projector(feats_fc)
        nce_loss = simclr_loss_func(z, indexes=targets, temperature=self.cfg.method_kwargs.temperature,)
        loss = nce_loss
        log_dict = {"train_nce_loss": nce_loss}

        if self.cfg.method_kwargs.classifier.enable:
            logits = torch.cat([out_i["logits"] for out_i in out], dim=0)
            cls_loss = F.cross_entropy(logits, targets, ignore_index=-1)
            loss = loss + cls_loss
            log_dict.update({"train_cls_loss": cls_loss})
            metrics = {"logits": logits.detach().cpu(), "targets": targets.detach().cpu()}
            self.train_step_outputs.append(metrics)

        self.log_dict(log_dict, on_epoch=True)

        return loss

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