from utils import get_args, get_dataset
from config import get_config

import models


def main():
    args = get_args()
    cfg = get_config()

    model = getattr(models, args.model)
    dataset = get_dataset(args, cfg)


if __name__ == "__main__":
    main()
