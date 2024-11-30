import torch, omegaconf
import pytorch_lightning as pl

from pathlib import Path
from nvidia.dali.plugin.pytorch import LastBatchPolicy

from solo.utils.misc import omegaconf_select
from solo.data.dali_wrap import Wrapper, PretrainWrapper
from solo.data.dali_transform_build import (PretrainPipelineBuilder, NormalPipelineBuilder,
                                            build_transform_pipeline_dali)


class PretrainDALIDataModule(pl.LightningDataModule):
    def __init__(self, cfg, train_data_path, val_data_path, transforms, wandb_path):
        super().__init__()
        self.wandb_path = wandb_path
        self.cfg = cfg
        self.val_data_path = Path(val_data_path)
        self.train_data_path = Path(train_data_path)
        self.transforms = transforms
        self.num_large_crops = cfg.data.num_large_crops
        self.num_workers: int = cfg.data.num_workers
        self.batch_size = cfg.optimizer.batch_size
        self.dali_device = cfg.dali.device
        self.balance = cfg.data.class_balance
        self.dataset = cfg.data.dataset

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        cfg.dali = omegaconf_select(cfg, "dali", {})
        cfg.dali.device = omegaconf_select(cfg, "dali.device", "gpu")
        return cfg

    def setup(self, stage = None):
        self.device_id = self.trainer.local_rank

        if torch.cuda.is_available() and self.dali_device == "gpu":
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self.device = torch.device("cpu")

        train_pipeline_builder = PretrainPipelineBuilder(
            self.train_data_path,
            batch_size=self.batch_size,
            transforms=self.transforms,
            device=self.dali_device,
            device_id=self.device_id,
            num_threads=self.num_workers,
            balance=self.balance,
            save_dir=self.wandb_path,
        )
        train_pipeline = train_pipeline_builder.pipeline(
            batch_size=train_pipeline_builder.batch_size,
            num_threads=train_pipeline_builder.num_threads,
            device_id=train_pipeline_builder.device_id,
            seed=train_pipeline_builder.seed,
        )
        train_pipeline.build()

        self.train_loader = PretrainWrapper(
            self.cfg,
            model_batch_size=self.batch_size,
            model_rank=self.device_id,
            model_device=self.device,
            dataset_size=train_pipeline_builder.data_len,
            conversion_map=None,
            pipelines=train_pipeline,
            output_map=([f"image{i}" for i in range(self.num_large_crops)] + ["label"] + ["filename"]),
            reader_name=None,
            last_batch_policy=LastBatchPolicy.DROP if self.cfg.data.drop_last else LastBatchPolicy.PARTIAL,
            auto_reset=True,
        )

        val_pipeline_builder = NormalPipelineBuilder(
            self.dataset,
            self.val_data_path,
            batch_size=self.batch_size,
            save_dir=self.wandb_path,
            device=self.dali_device,
            device_id=self.device_id,
            num_threads=self.num_workers,
            resize_size=self.cfg.augmentations[0].resize_size,
            crop_size=self.cfg.augmentations[0].crop_size,
        )
        val_pipeline = val_pipeline_builder.pipeline(
            batch_size=val_pipeline_builder.batch_size,
            num_threads=val_pipeline_builder.num_threads,
            device_id=val_pipeline_builder.device_id,
            seed=val_pipeline_builder.seed,
        )
        val_pipeline.build()

        self.val_loader = Wrapper(
            pipelines=val_pipeline,
            model_bacth_size=self.batch_size,
            dataset_size=val_pipeline_builder.data_len,
            output_map=["x", "label", "filename"],
            reader_name=None,
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


class ClassificationDALI():
    def __init__(self, cfg, test_data_path, bz=16):
        super().__init__()
        self.cfg = cfg
        self.device_id = 0
        if torch.cuda.is_available() and cfg.dali.device == "gpu":
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self.device = torch.device("cpu")

        test_pipeline_builder = NormalPipelineBuilder(
            cfg.data.dataset,
            test_data_path,
            batch_size=bz,
            device=cfg.dali.device,
            device_id=self.device_id,
            save_dir=None,
            num_threads=cfg.data.num_workers,
            resize_size=cfg.augmentations[0].resize_size,
            crop_size=cfg.augmentations[0].crop_size,
        )

        test_pipeline = test_pipeline_builder.pipeline(
            batch_size=test_pipeline_builder.batch_size,
            num_threads=test_pipeline_builder.num_threads,
            device_id=test_pipeline_builder.device_id,
            seed=test_pipeline_builder.seed,
        )
        test_pipeline.build()

        self.test_loader = Wrapper(
            pipelines=test_pipeline,
            model_bacth_size=cfg.optimizer.batch_size,
            dataset_size=test_pipeline_builder.data_len,
            output_map=["x", "label", "filename"],
            reader_name=None,
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True,
        )

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        cfg.dali = omegaconf_select(cfg, "dali", {})
        cfg.dali.device = omegaconf_select(cfg, "dali.device", "gpu")
        return cfg

    def test_dataloader(self):
        return self.test_loader