import os


def get_accuracy(model, dataset, transforms, args, cfg, top5=False):
    model.eval()

    dataset_root = os.path.join(
        cfg.data.root, os.path.join(*getattr(cfg.data, args.dataset).path.split("/"))
    )

    if getattr(cfg.data, args.dataset).params is not None:
        params = getattr(cfg.data, args.dataset).params
        return dataset.eval_model(
            model,
            dataset_root,
            *params,
            transforms=transforms,
            batch_size=args.batch_size,
            top5=top5,
            num_workers=args.workers,
        )
    else:
        return dataset.eval_model(
            model,
            root=dataset_root,
            transforms=transforms,
            batch_size=args.batch_size,
            top5=top5,
            num_workers=args.workers,
        )


def get_top1_accuracy(model, dataset, transforms, args, cfg):
    return get_accuracy(model, dataset, transforms, args, cfg, top5=False)


def get_top5_accuracy(model, dataset, transforms, args, cfg):
    return get_accuracy(model, dataset, transforms, args, cfg, top5=True)
