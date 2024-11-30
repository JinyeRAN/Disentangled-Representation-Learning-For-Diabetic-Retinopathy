import os
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali import pipeline_def

from pathlib import Path
import numpy as np
from solo.data.dali_external_source import ExternalInputGpuIterator
from solo.data.dali_transform import (RandomColorJitter, RandomGrayScaleConversion,
                                      RandomGaussianBlur, RandomSolarize, RandomRotate, RandomEqualize)
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

MEANS_N_STD = {
    "imagenet100": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    # 'uwf': ([0.265, 0.184, 0.005], [0.133, 0.103, 0.006])
    'uwf': ([0.3943, 0.3397, 0.0367], [0.1628, 0.1531, 0.0255])
}

class PretrainPipelineBuilder:
    def __init__(
        self, data_path, batch_size, device, transforms, balance, save_dir, device_id = 0, num_threads = 4, seed = 12):
        super().__init__()
        self.seed = seed + device_id
        self.device, data_path = device, Path(data_path)
        self.batch_size, self.num_threads, self.device_id = batch_size, num_threads, device_id

        # manually load files and labels
        labels = sorted(Path(entry.name) for entry in os.scandir(data_path) if entry.is_dir())
        data = [
            (data_path / label / file, label_idx)
            for label_idx, label in enumerate(labels)
            for file in sorted(os.listdir(data_path / label))
        ]
        files, labels = map(list, zip(*data))

        if not save_dir == None:
            root = os.path.join('./trained_models/DATA_INDEX', save_dir)
            if not os.path.exists(root): os.mkdir(root)
            np.save(os.path.join(root, 'train_img.npy'), np.array(files))
            np.save(os.path.join(root, 'train_lbl.npy'), np.array(files))

        self.data_len = len(files)
        self.eii = ExternalInputGpuIterator(files, labels, num_classes=5, batch_size=batch_size, balance=balance)

        decoder_device = "mixed" if self.device == "gpu" else "cpu"
        device_memory_padding = 211025920 if decoder_device == "mixed" else 0
        host_memory_padding = 140544512 if decoder_device == "mixed" else 0
        self.decode = ops.decoders.Image(
            device=decoder_device, output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
        )
        self.to_int64 = ops.Cast(dtype=types.INT64, device=device)
        self.transforms = transforms

    @pipeline_def(enable_conditionals=True)
    def pipeline(self):
        inputs, labels, filename = fn.external_source(source=self.eii, num_outputs=3, dtype=types.UINT8)
        # inputs, labels = fn.external_source(source=self.eii, num_outputs=2, dtype=types.UINT8)
        images = self.decode(inputs)
        crops = self.transforms(images)

        if self.device == "gpu":
            labels = labels.gpu()
            filename = filename.gpu()
        labels = self.to_int64(labels)
        filename = self.to_int64(filename)

        return (*crops, labels, filename)


class NormalPipelineBuilder:
    def __init__(
        self, dataset, data_path, batch_size: int, resize_size: int, crop_size: int, save_dir:str,
        device: str, device_id: int = 0, num_threads: int = 4, seed: int = 12,
    ):

        super().__init__()
        mean, std = MEANS_N_STD.get(dataset)

        self.seed = seed + device_id
        self.device, self.device_id = device, device_id
        self.batch_size, self.num_threads = batch_size, num_threads

        labels = sorted(Path(entry.name) for entry in os.scandir(data_path) if entry.is_dir())
        data = [
            (data_path / label / file, label_idx)
            for label_idx, label in enumerate(labels)
            for file in sorted(os.listdir(data_path / label))
        ]
        files, labels = map(list, zip(*data))

        if not save_dir==None:
            root = os.path.join('./trained_models/DATA_INDEX', save_dir)
            if not os.path.exists(root): os.mkdir(root)
            np.save(os.path.join(root, 'test_img.npy'), np.array(files))
            np.save(os.path.join(root, 'test_lbl.npy'), np.array(files))

        self.data_len = len(files)
        self.eii = ExternalInputGpuIterator(files, labels, num_classes=5, batch_size=batch_size, balance=False)

        decoder_device = "mixed" if self.device == "gpu" else "cpu"
        device_memory_padding = 211025920 if decoder_device == "mixed" else 0
        host_memory_padding = 140544512 if decoder_device == "mixed" else 0
        self.decode = ops.decoders.Image(
            device=decoder_device, output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
        )

        self.resize = ops.Resize(device=self.device,size=(resize_size, resize_size),interp_type=types.INTERP_CUBIC,)
        self.cmn = ops.CropMirrorNormalize(
            device=self.device, dtype=types.FLOAT, output_layout=types.NCHW, crop=(crop_size, crop_size),
            mean=[v * 255 for v in mean], std=[v * 255 for v in std],
        )

        self.to_int64 = ops.Cast(dtype=types.INT64, device=device)

    @pipeline_def
    def pipeline(self):
        inputs, labels, filename = fn.external_source(source=self.eii, num_outputs=3, dtype=types.UINT8)
        # inputs, labels = fn.external_source(source=self.eii, num_outputs=2, dtype=types.UINT8)
        images = self.decode(inputs)

        images = self.resize(images)
        images = self.cmn(images)

        if self.device == "gpu":
            labels = labels.gpu()
            filename = filename.gpu()
        labels = self.to_int64(labels)
        filename = self.to_int64(filename)
        return (images, labels, filename)


def build_transform_pipeline_dali(dataset, cfg, dali_device):
    mean, std = MEANS_N_STD.get(dataset)

    augmentations = []
    if cfg.rrc.enabled:
        augmentations.append(
            ops.RandomResizedCrop(
                device=dali_device, size=cfg.crop_size, interp_type=types.INTERP_CUBIC,
                random_area=(cfg.rrc.crop_min_scale, cfg.rrc.crop_max_scale),)
        )
    else:
        augmentations.append(
            ops.Resize(device=dali_device, size=(cfg.crop_size, cfg.crop_size),interp_type=types.INTERP_CUBIC,)
        )

    if cfg.color_jitter.prob:
        augmentations.append(
            RandomColorJitter(
                brightness=cfg.color_jitter.brightness, contrast=cfg.color_jitter.contrast,
                saturation=cfg.color_jitter.saturation, hue=cfg.color_jitter.hue,
                prob=cfg.color_jitter.prob, device=dali_device,
            )
        )

    if cfg.grayscale.prob:
        augmentations.append(RandomGrayScaleConversion(prob=cfg.grayscale.prob, device=dali_device))

    if cfg.gaussian_blur.prob:
        augmentations.append(RandomGaussianBlur(prob=cfg.gaussian_blur.prob, device=dali_device))

    if cfg.solarization.prob:
        augmentations.append(RandomSolarize(prob=cfg.solarization.prob))

    if cfg.equalization.prob:
        augmentations.append(RandomEqualize(prob=cfg.equalization.prob, device=dali_device))

    if cfg.rotate.prob:
        augmentations.append(RandomRotate(prob=cfg.rotate.prob, angle1=cfg.rotate.angle1,
                                          angle2=cfg.rotate.angle2, device=dali_device))

    coin = None
    if cfg.horizontal_flip.prob: coin = ops.random.CoinFlip(probability=cfg.horizontal_flip.prob)

    cmn = ops.CropMirrorNormalize(
        device=dali_device, dtype=types.FLOAT, output_layout=types.NCHW,
        mean=[v * 255 for v in mean], std=[v * 255 for v in std],
    )

    class AugWrapper:
        def __init__(self, augmentations, cmn, coin) -> None:
            self.augmentations, self.cmn, self.coin = augmentations, cmn, coin

        def __call__(self, images):
            for aug in self.augmentations: images = aug(images)

            if self.coin: images = self.cmn(images, mirror=self.coin())
            else: images = self.cmn(images)
            return images

    return AugWrapper(augmentations=augmentations, cmn=cmn, coin=coin)