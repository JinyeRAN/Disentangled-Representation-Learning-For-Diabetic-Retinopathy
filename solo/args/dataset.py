from argparse import ArgumentParser
from pathlib import Path


def dataset_args(parser: ArgumentParser):
    SUPPORTED_DATASETS = ["imagenet100", 'uwf']

    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, type=str, required=True)

    # dataset path
    parser.add_argument("--train_data_path", type=Path, required=True)
    parser.add_argument("--val_data_path", type=Path, default=None)
    parser.add_argument(
        "--data_format", default="image_folder", choices=["image_folder", "dali", "h5"]
    )


def augmentations_args(parser: ArgumentParser):
    # cropping
    parser.add_argument("--num_crops_per_aug", type=int, default=[2], nargs="+")

    # color jitter
    parser.add_argument("--brightness", type=float, required=True, nargs="+")
    parser.add_argument("--contrast", type=float, required=True, nargs="+")
    parser.add_argument("--saturation", type=float, required=True, nargs="+")
    parser.add_argument("--hue", type=float, required=True, nargs="+")
    parser.add_argument("--color_jitter_prob", type=float, default=[0.8], nargs="+")

    # other augmentation probabilities
    parser.add_argument("--gray_scale_prob", type=float, default=[0.2], nargs="+")
    parser.add_argument("--horizontal_flip_prob", type=float, default=[0.5], nargs="+")
    parser.add_argument("--gaussian_prob", type=float, default=[0.5], nargs="+")
    parser.add_argument("--solarization_prob", type=float, default=[0.0], nargs="+")
    parser.add_argument("--equalization_prob", type=float, default=[0.0], nargs="+")

    # cropping
    parser.add_argument("--crop_size", type=int, default=[224], nargs="+")
    parser.add_argument("--min_scale", type=float, default=[0.08], nargs="+")
    parser.add_argument("--max_scale", type=float, default=[1.0], nargs="+")

    # debug
    parser.add_argument("--debug_augmentations", action="store_true")