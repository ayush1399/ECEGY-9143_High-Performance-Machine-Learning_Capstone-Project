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
argparser.add_argument("--batch_size", type=int, default=128)
argparser.add_argument("--workers", type=int, default=8)
argparser.add_argument("--eval_mode", choices=["acc", "perf"], default="acc")
argparser.add_argument("--silent", action="store_true")
argparser.add_argument("--top5", action="store_true")


def get_args():
    return argparser.parse_args()


def get_dataset(args, cfg):
    transform = get_transform(args.dataset)

    dataset = getattr(datasets, args.dataset)
    dataset_root = os.path.join(
        cfg.data.root, os.path.join(*getattr(cfg.data, args.dataset).path.split("/"))
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


def pretty_print_perf(inference_time, throughput, args, cfg):
    print("=" + "*=" * 12)
    print(f"MODEL: {args.model: <10} DATASET: {args.dataset}")
    print(f"Batch Size: {args.batch_size: <10} Workers: {args.workers}")
    print(f"Config params: {getattr(cfg.data, args.dataset).params}")
    print(f"Inference Time: {inference_time:.3f} ms")
    print(f"Throughput: {throughput:.3f} images/s")
    print("=" + "*=" * 12)
    print()


def pretty_print_acc(acc, args, cfg, dataset):
    print("=" + "*=" * 12)
    print(f"MODEL: {args.model: <10} DATASET: {args.dataset}")
    print(f"Config params: {getattr(cfg.data, args.dataset).params}")
    dataset.print_acc(acc, args.top5)
    print("=" + "*=" * 12)
    print()
