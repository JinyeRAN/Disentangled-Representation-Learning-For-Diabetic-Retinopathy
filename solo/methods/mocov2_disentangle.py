from typing import Any, Dict, List, Sequence

import torch
import random
import omegaconf
import torch.nn as nn
import torch.nn.functional as F

from solo.methods.base import BaseMethod
from solo.utils.test_utils import valid_core
from solo.methods.utils import BatchNorm1d, Scaler, random_select, WarmStartGradientScaleLayer


class MocoDisentangle(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig, Pure: bool=False):
        super().__init__(cfg)
        self.cfg = cfg

        ####################### backbone #########################
        self.backbone = self._BACKBONES[cfg.backbone.name](cfg.method)
        self.backbone.fc, backbone_dim = nn.Identity(), self.backbone.inplanes

        if not Pure:
            if cfg.backbone.weight.enable:
                weight = torch.load(cfg.backbone.weight.root)['state_dict']
                self.load_state_dict(weight, False)

            self.cfm = cfg.method_kwargs
            self.hps = cfg.method_kwargs.disentangle.hparams

            if self.cfm.disentangle.enable:
                ######################### coder #########################
                enc_layers, encp_layers = self.cfm.disentangle.enc_layers.split(), self.cfm.disentangle.encp_layers.split()
                enc_kernel, encoder = cfg.method_kwargs.disentangle.encp_kernel.split(), []

                for i in range(len(enc_layers)):
                    num_hidden, pre_hidden = int(enc_layers[i]), int(enc_layers[i - 1])
                    if i == 0: encoder.extend([nn.Linear(backbone_dim, num_hidden), nn.ReLU()])
                    else: encoder.extend([nn.Dropout(p=0.3), nn.Linear(pre_hidden, num_hidden), nn.ReLU()])
                self.encoder = nn.Sequential(*encoder)

                self.linear = nn.Linear(int(enc_layers[-1]), self.cfm.disentangle.z_dim * 4)
                self.scale_sps, self.scale_cls = Scaler(self.cfm.disentangle.z_dim), Scaler(self.cfm.disentangle.z_dim)

                dec_layers, decp_layers = self.cfm.disentangle.dec_layers.split(), self.cfm.disentangle.decp_layers.split()
                dec_kernel, decoder = cfg.method_kwargs.disentangle.encp_kernel.split(), []

                for i in range(len(dec_layers)):
                    num_hidden, pre_hidden = int(dec_layers[i]), int(dec_layers[i - 1])
                    if i == 0: decoder.extend([nn.Linear(self.cfm.disentangle.z_dim, num_hidden), nn.ReLU()])
                    else: decoder.extend([nn.Linear(pre_hidden, num_hidden), nn.ReLU()])
                    if i == len(dec_layers)-1: decoder.extend([nn.Linear(num_hidden, backbone_dim), nn.ReLU()])
                self.decoder = nn.Sequential(*decoder)

                ######################### loss #########################
                self.register_buffer('cstn', torch.ones(1) * 0.5)
                self.register_parameter('casual_mu', nn.Parameter(torch.randn(self.cfm.classifier.classes - 1, self.cfm.disentangle.z_dim, requires_grad=True)))
                self.register_parameter('casual_sg', nn.Parameter(torch.randn(self.cfm.classifier.classes - 1, self.cfm.disentangle.z_dim, requires_grad=True)))
            else:
                enc_layers, encp_layers = self.cfm.disentangle.enc_layers.split(), self.cfm.disentangle.encp_layers.split()
                enc_kernel, encoder = cfg.method_kwargs.disentangle.encp_kernel.split(), []

                for i in range(len(enc_layers)):
                    num_hidden, pre_hidden = int(enc_layers[i]), int(enc_layers[i - 1])
                    if i == 0:
                        encoder.extend([nn.Linear(backbone_dim, num_hidden), nn.ReLU()])
                    else:
                        encoder.extend([nn.Dropout(p=0.3), nn.Linear(pre_hidden, num_hidden), nn.ReLU()])
                self.encoder = nn.Sequential(*encoder)

                self.linear = nn.Linear(int(enc_layers[-1]), self.cfm.disentangle.z_dim * 2)
                self.scale_cls = Scaler(self.cfm.disentangle.z_dim)

            ###################### classifier ######################
            if self.cfm.classifier.enable:
                dims = self.cfm.disentangle.z_dim if self.cfm.classifier.type=='cls_mu' else backbone_dim
                if self.cfm.classifier.type_C: dims = dims * 2
                self.classifier: nn.Module =  nn.Linear(dims, self.cfm.classifier.classes)
                self.valid_step_outputs, self.train_step_outputs = [], []

    @property
    def learnable_params(self) -> List[dict]:
        learnable_params = []
        if self.cfm.disentangle.enable:
            learnable_params.extend([
                {"name": "cmu", "params": self.casual_mu, },
                {"name": "csg", "params": self.casual_sg, },
                {"name": "lnr", "params": self.linear.parameters(), },
                {"name": "dcd", "params": self.decoder.parameters(), },
                {"name": "ecd", "params": self.encoder.parameters(), },
                {"name": "slrs", "params": self.scale_sps.parameters(), },
                {"name": "slrc", "params": self.scale_cls.parameters(), },
            ])
        else:
            learnable_params.extend([
                {"name": "lnr", "params": self.linear.parameters(), },
                {"name": "ecd", "params": self.encoder.parameters(), },
                {"name": "slrc", "params": self.scale_cls.parameters(), },
            ])
        if not self.cfg.backbone.weight.enable:
            learnable_params.append(
                {"name": "bkb", "params": self.backbone.parameters(),}
            )
        if self.cfm.classifier.enable:
            learnable_params.append(
                {"name": "cls", "params": self.classifier.parameters(), "lr": self.classifier_lr, "weight_decay": 0, }
            )

        return learnable_params

    def random_select(self, lbl_cls):
        if lbl_cls == 0:
            flag, step = True, random.randint(1, 4)
        elif lbl_cls == 4:
            flag, step = False, random.randint(1, 4)
        elif lbl_cls == 1:
            if random.random() > 0.5:  flag, step = True, random.randint(1, 3)
            else: flag, step = False, 1
        elif lbl_cls == 3:
            if random.random() > 0.5: flag, step = True, 1
            else: flag, step = False, random.randint(1, 3)
        else:
            if random.random() > 0.5: flag = True
            else: flag = False
            step = random.randint(1, 2)
        return flag, step

    def normal_dist(self, mu1, sg1, mu2, sg2):
        if len(mu1.shape)==1 and len(mu2.shape)==1:
            mu1, sg1 = mu1.unsqueeze(0), sg1.unsqueeze(0)
            mu2, sg2 = mu2.unsqueeze(0), sg2.unsqueeze(0)
        elif len(mu1.shape)<len(mu2.shape):
            mu1 = mu1.unsqueeze(0).repeat(mu2.shape[0], 1)
            sg1 = sg1.unsqueeze(0).repeat(sg2.shape[0], 1)
        elif len(mu1.shape) == len(mu2.shape):
            pass
        else: raise

        crt_sg_, csl_sg_ = torch.exp(sg1), torch.exp(sg2)
        distr = torch.sum(
            torch.pow(mu1 - mu2, 2) + crt_sg_ + csl_sg_ - 2 * torch.sqrt(
                torch.sqrt(crt_sg_).mul(csl_sg_).mul(torch.sqrt(crt_sg_))
            ), dim=1)
        return torch.mean(distr)

    def cls_dstl(self, mu_cls, sg_cls, mu_sps, sg_sps, lbl):
        disentangle_loss, unique_cls = torch.zeros(1).to(mu_cls.device), torch.unique(lbl)
        A, S, C = torch.zeros(1).to(mu_cls.device), torch.zeros(1).to(mu_cls.device), torch.zeros(1).to(mu_cls.device)

        for lbl_cls in unique_cls:
            # ZC = self.normal_dist(mu_cls[lbl == lbl_cls], sg_cls[lbl == lbl_cls], mu_sps[lbl == lbl_cls], sg_sps[lbl == lbl_cls])
            ZC = self.normal_dist(
                torch.mean(mu_cls[lbl == lbl_cls], dim=0),
                torch.mean(sg_cls[lbl == lbl_cls], dim=0),
                torch.mean(mu_sps[lbl == lbl_cls], dim=0).detach(),
                torch.mean(sg_sps[lbl == lbl_cls], dim=0).detach()
            ) * 0.2
            mu_cls_genert = torch.mean(mu_cls[lbl == lbl_cls], dim=0)
            sg_cls_genert = torch.mean(sg_cls[lbl == lbl_cls], dim=0)

            flag, step = self.random_select(lbl_cls)
            if flag:
                mu_cls_target = mu_cls[lbl == (lbl_cls + step)]
                sg_cls_target = sg_cls[lbl == (lbl_cls + step)]

                # mu_sps_target = mu_sps[lbl == (lbl_cls + step)]
                # sg_sps_target = sg_sps[lbl == (lbl_cls + step)]

                for j in range(step): mu_cls_genert = mu_cls_genert + self.casual_mu[lbl_cls + j]
                for j in range(step): sg_cls_genert = sg_cls_genert + self.casual_sg[lbl_cls + j]

            else:
                mu_cls_target = mu_cls[lbl == (lbl_cls - step)]
                sg_cls_target = sg_cls[lbl == (lbl_cls - step)]

                # mu_sps_target = mu_sps[lbl == (lbl_cls - step)]
                # sg_sps_target = sg_sps[lbl == (lbl_cls - step)]

                for j in range(step): mu_cls_genert = mu_cls_genert - self.casual_mu[lbl_cls - j - 1]
                for j in range(step): sg_cls_genert = sg_cls_genert - self.casual_sg[lbl_cls - j - 1]

            ZA = self.normal_dist(mu_cls_genert, sg_cls_genert, mu_cls_target.detach(), sg_cls_target.detach()) * 5
            # ZS = self.normal_dist(mu_cls_genert, sg_cls_genert, mu_sps_target, sg_sps_target)
            # distr = -torch.log((ZC + ZS) / (ZC + ZS + ZA))
            distr = -torch.log(ZC / (ZC + ZA))
            disentangle_loss = disentangle_loss + distr
            A = A + ZA
            S = 0 # S + ZS
            C = C + ZC
        return disentangle_loss / len(unique_cls), A / len(unique_cls), C / len(unique_cls), S / len(unique_cls)

    def train_btneck(self, feats_map, output):
        fusion = self.linear(self.encoder(feats_map))
        cls_mu, cls_sigma = torch.chunk(fusion, 2, dim=1)
        cls_mu, cls_sigma = self.scale_cls(cls_mu, cls_sigma)
        output.update(
            {
                'cls_mu': cls_mu,
                'cls_sigma': cls_sigma,
            }
        )
        return output

    def train_disentangle(self, feats_map, output):
        fusion = self.linear(self.encoder(feats_map))
        cls_mu, cls_sigma, sps_mu, sps_sigma = torch.chunk(fusion, 4, dim=1)
        cls_mu, cls_sigma = self.scale_cls(cls_mu, cls_sigma)
        sps_mu, sps_sigma = self.scale_sps(sps_mu, sps_sigma)

        Z_cls = torch.randn_like(cls_sigma) * torch.exp(0.5 * cls_sigma) + cls_mu
        Z_sps = torch.randn_like(sps_sigma) * torch.exp(0.5 * sps_sigma) + sps_mu

        _, L = Z_cls.shape
        for indx in [0,2,4,6]:
            Z_sps = self.decoder[indx+1](self.decoder[indx](Z_sps))
            Z_sps = Z_sps + F.interpolate(Z_cls.unsqueeze(1), scale_factor=Z_sps.shape[1]//L).squeeze(1)
        for indxs in range(8,14):
            Z_sps = self.decoder[indxs](Z_sps)
        output.update(
            {
                'cls_mu': cls_mu,
                'cls_sigma': cls_sigma,
                'sps_mu': sps_mu,
                'sps_sigma': sps_sigma,
                'feats_rs': Z_sps,
            }
        )
        return output

    @torch.no_grad()
    def obtain_feats(self, X:torch.tensor):
        self.backbone.eval()
        feats_fc = self.backbone(X)
        return feats_fc

    def forward(self, X, target):
        if not self.no_channel_last: X = X.to(memory_format=torch.channels_last)

        feats = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(self.obtain_feats(X)), 1)
        output = {'feats': feats, 'target': target}

        if self.cfm.disentangle.enable: output = self.train_disentangle(feats, output)
        else: output = self.train_btneck(feats, output)

        if self.cfm.classifier.enable:
            if self.cfm.classifier.type_C: key = torch.cat([output[self.cfm.classifier.type], output[self.cfm.classifier.type_C]], dim=1)
            else: key = output[self.cfm.classifier.type]
            if self.cfm.classifier.grad_cut: logit = self.classifier(key.detach())
            else: logit = self.classifier(key)
            output.update({"logits":logit, })

        return output

    def kl_dt(self, mu, sg, dim):
        kl = torch.mean(self.cstn * torch.sum(torch.exp(sg) + torch.pow(mu, 2) - 1 - sg, dim=dim))
        return kl

    def training_step(self, batch: Sequence[Any]):
        X, targets, index = batch
        X = [X] if isinstance(X, torch.Tensor) else X

        out = [self.forward(x_i, targets) for x_i in X]
        out = {k: torch.cat([out[k] for out in out], dim=0) for k in out[0].keys()}

        log_dict, loss = {}, .0
        if self.cfm.classifier.enable:
            cls_logit = F.cross_entropy(out["logits"], out['target'], ignore_index=-1)
            log_dict.update({"train_classifier_loss": cls_logit})
            self.train_step_outputs.append({"logits": out["logits"].detach().cpu(), "targets": out['target'].detach().cpu()})
            loss = self.hps.cls * cls_logit + loss

        if self.cfm.disentangle.enable:
            rcs = torch.mean(self.cstn * torch.sum(torch.pow(out['feats_rs'] - out['feats'], 2), dim=(1)))
            kl_cls = self.kl_dt(out['cls_sigma'], out['cls_mu'], dim=1)
            kl_sps = self.kl_dt(out['sps_sigma'], out['sps_mu'], dim=1)
            kl_cas = self.kl_dt(self.casual_sg, self.casual_mu, dim=1)
            dt_cls, A, C, S = self.cls_dstl(out['cls_mu'], out['cls_sigma'], out['sps_mu'], out['sps_sigma'], out['target'])
            loss = loss + rcs + self.hps.kl_sps * kl_sps + self.hps.kl_cls * kl_cls  + self.hps.kl_dt * kl_cas + self.hps.dt_cls * dt_cls
            log_dict.update(
                {
                    "train_rcs_loss": rcs,
                    "train_KL_cls_loss": kl_cls,
                    "train_KL_sps_loss": kl_sps,
                    "train_KL_cas_loss": kl_cas,
                    "train_DT_cls_loss": dt_cls,
                    "train_A_loss": A,
                    "train_S_loss": S,
                    "train_C_loss": C,
                }
            )

        self.log_dict(log_dict, on_epoch=True)
        return loss

    def on_train_epoch_end(self) -> None:
        if self.cfm.classifier.enable:
            if not len(self.train_step_outputs)==0:
                logits  = torch.cat([i['logits']  for i in self.train_step_outputs], dim=0)
                targets = torch.cat([i['targets'] for i in self.train_step_outputs], dim=0)
                train_acc = valid_core(logits, targets)
                self.log_dict({'train_acc_avg':train_acc,})
                self.train_step_outputs.clear()

    def validation_step(self, batch, batch_id):
        X, targets, index = batch
        out = self.forward(X, targets)

        if self.cfm.classifier.enable:
            metrics = {'logits':out['logits'].detach().cpu(), 'targets':out['target'].detach().cpu(),}
            self.valid_step_outputs.append(metrics)

    def on_validation_epoch_end(self):
        if self.cfm.classifier.enable:
            if not len(self.valid_step_outputs) == 0:
                logits  = torch.cat([i['logits'] for i in self.valid_step_outputs], dim=0)
                targets = torch.cat([i['targets'] for i in self.valid_step_outputs], dim=0)
                valid_acc = valid_core(logits, targets)
                self.log_dict({'valid_acc_avg': valid_acc})
                self.valid_step_outputs.clear()