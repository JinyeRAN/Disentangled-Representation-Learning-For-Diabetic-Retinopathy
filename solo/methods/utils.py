import torch, random
import torch.nn as nn
from torch.autograd import Function
from typing import Optional, Any, Tuple
import numpy as np


class BatchNorm1d(nn.Module):
    def __init__(self, dim):
        super(BatchNorm1d, self).__init__()
        self.BatchNorm1d = nn.BatchNorm1d(dim)

    def __call__(self, x):
        if x.size(0) == 1:
            x = torch.cat((x,x), 0)
            x = self.BatchNorm1d(x)[:1]
        else:
            x = self.BatchNorm1d(x)
        return x


class Scaler(nn.Module):
    def __init__(self, dim, tau=0.5):
        super(Scaler, self).__init__()
        self.tau = tau
        self.act = nn.Sigmoid()
        self.register_parameter('scale', nn.Parameter(torch.randn(1), requires_grad=True))
        self.mu_bn, self.sg_bn = nn.BatchNorm1d(dim, affine=False), nn.BatchNorm1d(dim, affine=False)

    def calculate_scale(self, inputs, mode):
        if mode == 'positive':
            scale = self.tau + (1 - self.tau) * self.act(self.scale)
        else:
            scale = (1 - self.tau) * self.act(-self.scale)
        return inputs * torch.sqrt(scale)

    def forward(self, mu, logvar):
        mu = self.mu_bn(mu)
        logvar = self.sg_bn(logvar)

        mu = self.calculate_scale(mu, mode='positive')
        logvar = self.calculate_scale(logvar, mode='negative')
        return mu, logvar

def random_select(lbl_cls):
    if lbl_cls == 0:
        flag = True
    elif lbl_cls == 4:
        flag = False
    else:
        if random.random() > 0.5:
            flag = True
        else:
            flag = False
    return flag


class GradientScaleFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output * ctx.coeff, None


def WarmStartGradientScaleLayer(input, global_step, max_step, alpha) -> torch.Tensor:
    coeff = float(2.0 / (1.0 + np.exp(- alpha * global_step / max_step)) - 1)
    return GradientScaleFunction.apply(input, coeff)

#
# import torch
# from pytorch_lightning import seed_everything
# seed_everything(0)
#
# target = torch.tensor([1,])
# data = torch.randn(1, 4)
# model1 = nn.Linear(4, 2)
# model2 = nn.Linear(2, 1)
# a = model1(data)
# # a = WarmStartGradientScaleLayer(a, 10, 100, 0.1)
# output = model2(a)
# loss = target - output
# loss.backward()
# print(model1.weight.grad)
