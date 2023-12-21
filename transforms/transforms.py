from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomResizedCrop,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
import torchvision.models as M

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
crop = 224
resize = 256

imagenet1k_train_transform = Compose(
    [
        RandomResizedCrop(crop),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ]
)

imagenet1k_eval_transform = Compose(
    [
        Resize(resize),
        CenterCrop(crop),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ]
)


def get_transform(model, dataset):
    
    if model == "ViT":
        return M.ViT_B_32_Weights.IMAGENET1K_V1.transforms()
    if model == "Swin_V2":
        return M.Swin_V2_B_Weights.IMAGENET1K_V1.transforms()
    if model == "ResNet":
        return M.ResNet152_Weights.IMAGENET1K_V1.transforms()
    
    if (
        dataset == "ImagenetA"
        or dataset == "ImagenetC"
        or dataset == "ImagenetR"
        or dataset == "ImagenetV2"
        or dataset == "Imagenet1KVal"
        or dataset == "Imagenet1KTest"
    ):
        return imagenet1k_eval_transform
    elif dataset == "Imagenet1KTrain":
        return imagenet1k_train_transform
    raise ValueError(f"Unknown dataset {dataset}")
