import os

import omegaconf
from omegaconf import OmegaConf
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import ModelCheckpoint as Checkpointer
from solo.utils.misc import omegaconf_select
from solo.data.dali_dataloader import PretrainDALIDataModule

_N_CLASSES_PER_DATASET = {"imagenet100": 5, "uwf":5}
_SUPPORTED_DATASETS = ["imagenet100", "uwf"]


def add_and_assert_dataset_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    assert not OmegaConf.is_missing(cfg, "data.dataset")
    assert not OmegaConf.is_missing(cfg, "data.train_path")

    assert cfg.data.dataset in _SUPPORTED_DATASETS

    # if validation path is not available, assume that we want to skip eval
    cfg.data.val_path = omegaconf_select(cfg, "data.val_path", None)
    cfg.data.format = omegaconf_select(cfg, "data.format", "image_folder")
    cfg.debug_augmentations = omegaconf_select(cfg, "debug_augmentations", False)

    return cfg


def add_and_assert_wandb_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    cfg.wandb = omegaconf_select(cfg, "wandb", {})
    cfg.wandb.enabled = omegaconf_select(cfg, "wandb.enabled", False)
    cfg.wandb.entity = omegaconf_select(cfg, "wandb.entity", None)
    cfg.wandb.project = omegaconf_select(cfg, "wandb.project", "solo-learn")
    cfg.wandb.offline = omegaconf_select(cfg, "wandb.offline", False)

    return cfg


def add_and_assert_lightning_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    cfg.seed = omegaconf_select(cfg, "seed", 0)
    cfg.resume_from_checkpoint = omegaconf_select(cfg, "resume_from_checkpoint", None)
    cfg.strategy = omegaconf_select(cfg, "strategy", None)

    return cfg


def parse_cfg(cfg: omegaconf.DictConfig):
    # default values for checkpointer
    cfg = Checkpointer.add_and_assert_specific_cfg(cfg)

    # default values for auto_resume
    cfg = AutoResumer.add_and_assert_specific_cfg(cfg)

    # default values for dali
    cfg = PretrainDALIDataModule.add_and_assert_specific_cfg(cfg)

    # assert dataset parameters
    cfg = add_and_assert_dataset_cfg(cfg)

    # default values for wandb
    cfg = add_and_assert_wandb_cfg(cfg)

    # default values for pytorch lightning stuff
    cfg = add_and_assert_lightning_cfg(cfg)

    # extra processing
    if cfg.data.dataset in _N_CLASSES_PER_DATASET:
        cfg.data.num_classes = _N_CLASSES_PER_DATASET[cfg.data.dataset]
    else:
        cfg.data.num_classes = max(1, sum(entry.is_dir() for entry in os.scandir(cfg.data.train_path)),)

    # find number of big/small crops
    big_size = cfg.augmentations[0].crop_size
    num_large_crops = num_small_crops = 0
    for pipeline in cfg.augmentations:
        if big_size == pipeline.crop_size: num_large_crops += pipeline.num_crops
        else: num_small_crops += pipeline.num_crops
    cfg.data.num_large_crops = num_large_crops
    cfg.data.num_small_crops = num_small_crops

    # extra optimizer kwargs
    cfg.optimizer.kwargs = omegaconf_select(cfg, "optimizer.kwargs", {})
    if cfg.optimizer.name == "sgd":
        cfg.optimizer.kwargs.momentum = omegaconf_select(cfg, "optimizer.kwargs.momentum", 0.9)
    elif cfg.optimizer.name == "lars":
        cfg.optimizer.kwargs.momentum = omegaconf_select(cfg, "optimizer.kwargs.momentum", 0.9)
        cfg.optimizer.kwargs.eta = omegaconf_select(cfg, "optimizer.kwargs.eta", 1e-3)
        cfg.optimizer.kwargs.clip_lr = omegaconf_select(cfg, "optimizer.kwargs.clip_lr", False)
    elif cfg.optimizer.name == "adamw":
        cfg.optimizer.kwargs.betas = omegaconf_select(cfg, "optimizer.kwargs.betas", [0.9, 0.999])

    return cfg
