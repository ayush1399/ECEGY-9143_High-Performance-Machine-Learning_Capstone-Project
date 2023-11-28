from ray import get
from transforms import get_transform

import argparse
import datasets
import os

eval_datasets = [
    "Imagenet1KTest",
    "Imagenet1KVal",
    "ImagenetA",
    "ImagenetC",
    "ImagenetR",
    "ImagenetV2",
]

eval_models = ["RPN", "RPN-P", "RPN-PQ", "RPN-PQ-EE", "ViT", "ResNet"]

argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset", choices=eval_datasets, required=True)
argparser.add_argument("--model", choices=eval_models, required=True)
argparser.add_argument("--batch-size", type=int, default=128)
argparser.add_argument("--workers", type=int, default=8)
argparser.add_argument("--eval-mode", choices=["acc", "inference"])


def get_args():
    return argparser.parse_args()


def get_dataset(args, cfg):
    transform = get_transform(args.dataset)

    dataset = getattr(datasets, args.dataset)
    dataset_root = os.path.join(
        cfg.data.root, os.path.join(getattr(cfg.data, args.dataset).path.split("/"))
    )

    if getattr(cfg.data, args.dataset).params is not None:
        params = getattr(cfg.data, args.dataset).params
        return dataset(
            dataset_root,
            *params,
            transform=transform,
        )

    return dataset(
        dataset_root,
        transform=transform,
    )
