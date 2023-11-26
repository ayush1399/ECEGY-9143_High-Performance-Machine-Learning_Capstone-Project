import argparse
import datasets
import models

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


def main():
    args = argparser.parse_args()

    model = getattr(models, args.model)
    dataset = getattr(datasets, args.dataset)


if __name__ == "__main__":
    main()
