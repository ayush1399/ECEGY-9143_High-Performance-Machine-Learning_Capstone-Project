from torchvision.models import resnet152, ResNet152_Weights


def ResNet(weights=ResNet152_Weights.IMAGENET1K_V1, progress=False, **kwargs):
    return resnet152(weights=weights, progress=progress, **kwargs)


import ViT
import RPN
import RPN_P
import RPN_PQ
import RPN_PQ_EE

__all__ = ["ResNet", "RPN", "RPN_P", "RPN_PQ", "RPN_PQ_EE", "ViT"]
