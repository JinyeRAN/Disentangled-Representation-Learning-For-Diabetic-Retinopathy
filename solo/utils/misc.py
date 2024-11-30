import torch
from omegaconf import OmegaConf


def make_contiguous(module):
    with torch.no_grad():
        for param in module.parameters():
            param.set_(param.contiguous())

def omegaconf_select(cfg, key, default=None):
    """Wrapper for OmegaConf.select to allow None to be returned instead of 'None'."""
    value = OmegaConf.select(cfg, key, default=default)
    if value == "None":
        return None
    return value
