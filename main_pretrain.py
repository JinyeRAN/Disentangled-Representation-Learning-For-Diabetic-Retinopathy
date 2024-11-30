import os, sys, torch, hydra, inspect, warnings
warnings.filterwarnings('ignore')

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from solo.methods import METHODS
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import ModelCheckpoint
from solo.utils.misc import make_contiguous

from solo.args.pretrain import parse_cfg
from solo.data.dali_dataloader import PretrainDALIDataModule as PDDM
from solo.data.dali_transform_build import build_transform_pipeline_dali as butp_dali
from solo.data.utils import NCropAugmentation as NCA
from solo.data.utils import FullTransformPipeline as FTP


@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)

    if (True if sys.gettrace() else False):
        cfg.wandb.offline = True

    seed_everything(cfg.seed)
    model = METHODS[cfg.method](cfg)
    make_contiguous(model)
    if not cfg.performance.disable_channel_last: model = model.to(memory_format=torch.channels_last)

    ckpt_path, wandb_run_id = None, None
    if cfg.auto_resume.enabled and cfg.resume_from_checkpoint is None:
        auto_resumer = AutoResumer(checkpoint_dir=os.path.join(cfg.checkpoint.dir, cfg.method))
        resume_from_checkpoint, wandb_run_id = auto_resumer.find_checkpoint(cfg)
        if resume_from_checkpoint is not None:
            print("Resuming from previous checkpoint that matches specifications:",f"'{resume_from_checkpoint}'",)
            ckpt_path = resume_from_checkpoint
    elif cfg.resume_from_checkpoint is not None:
        ckpt_path = cfg.resume_from_checkpoint
        del cfg.resume_from_checkpoint

    callbacks = []

    if cfg.checkpoint.enabled:
        ckpt = ModelCheckpoint(
            cfg, monitor=cfg.checkpoint.monitor,
            frequency=cfg.checkpoint.frequency, save_top_module=cfg.checkpoint.save_top_module,
            save_last_ckpt=cfg.checkpoint.save_last_ckpt, dirpath = os.path.join(cfg.checkpoint.dir, cfg.method),
        )
        callbacks.append(ckpt)

    if cfg.wandb.enabled:
        wandb_logger = WandbLogger(
            name=cfg.name, project=cfg.wandb.project, entity=cfg.wandb.entity,
            offline=cfg.wandb.offline, resume="allow" if wandb_run_id else None, id=wandb_run_id,
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))

        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    pipelines = []
    for aug_cfg in cfg.augmentations:
        pipelines.append(NCA(butp_dali(cfg.data.dataset, aug_cfg, dali_device=cfg.dali.device), aug_cfg.num_crops,))
    transform = FTP(pipelines)
    # saved_path = wandb_logger.version
    dali_datamodule = PDDM(cfg, train_data_path=cfg.data.train_path, val_data_path=cfg.data.val_path,
                           transforms=transform, wandb_path=None) # wandb_logger.version)

    trainer_kwargs = OmegaConf.to_container(cfg)
    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {name: trainer_kwargs[name] for name in valid_kwargs if name in trainer_kwargs}
    trainer_kwargs.update(
        {
            "logger": wandb_logger if cfg.wandb.enabled else None,
            "callbacks": callbacks,
            "strategy": cfg.strategy,
            "check_val_every_n_epoch": 1,
            "log_every_n_steps":  10,
            "num_sanity_val_steps": 0
        }
    )
    trainer = Trainer(**trainer_kwargs)

    try:
        from pytorch_lightning.loops import FitLoop

        class WorkaroundFitLoop(FitLoop):
            @property
            def prefetch_batches(self) -> int:  return 1

        trainer.fit_loop = WorkaroundFitLoop(trainer.fit_loop.min_epochs, trainer.fit_loop.max_epochs)
    except:
        pass

    trainer.fit(model, ckpt_path=ckpt_path, datamodule=dali_datamodule)


if __name__ == "__main__":
    main()
