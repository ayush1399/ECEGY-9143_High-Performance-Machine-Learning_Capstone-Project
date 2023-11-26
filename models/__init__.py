from torchvision.models import resnet152 as _resnet152

models = {
    "RPN": None,
    "RPN-P": None,
    "RPN-PQ": None,
    "RPN-PQ-EE": None,
    "ViT": None,
    "ResNet": _resnet152,
}

__all__ = ["models"]
