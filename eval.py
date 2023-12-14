from utils import get_args, get_dataset, pretty_print_perf
from benchmarking import get_performance, get_top1_accuracy, get_top5_accuracy
from torch.utils.data import DataLoader
from config import get_config

import models


def exec(args, cfg, model, dataset):
    if args.eval_mode == "perf":
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            shuffle=False,
        )

        inference_time, throughput = get_performance(
            model, dataloader, cfg.model.input_shape
        )

        if args.silent:
            print(f"Inference Time: {inference_time} \t Throughput: {throughput}")
        else:
            pretty_print_perf(inference_time, throughput, args, cfg)

    elif args.eval_mode == "acc":
        dataset, transforms = dataset
        if args.top5:
            acc = get_top5_accuracy(model, dataset, transforms, args, cfg)
        else:
            acc = get_top1_accuracy(model, dataset, transforms, args, cfg)
        pretty_print_perf(acc, args, cfg, dataset)
    else:
        raise NotImplementedError


def main():
    args = get_args()
    cfg = get_config()

    model = getattr(models, args.model)()
    dataset = get_dataset(args, cfg)

    exec(args, cfg, model, dataset)


if __name__ == "__main__":
    main()
