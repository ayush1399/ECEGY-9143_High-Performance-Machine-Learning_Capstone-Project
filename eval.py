from utils import get_args, get_dataset, pretty_print_perf
from benchmarking import get_inference_time
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
        inference_time = get_inference_time(model, dataloader, cfg.model.input_shape)
        if args.silent:
            print(f"Inference Time: {inference_time}")
        else:
            pretty_print_perf(inference_time, args, cfg)

    elif args.eval_mode == "acc":
        pass

    raise NotImplementedError


def main():
    args = get_args()
    cfg = get_config()

    model = getattr(models, args.model)
    dataset = get_dataset(args, cfg)

    exec(args, cfg, model, dataset)


if __name__ == "__main__":
    main()
