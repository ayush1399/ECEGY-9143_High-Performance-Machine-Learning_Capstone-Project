import torchvision.models as M

import RPN
import RPN_P
import RPN_PQ
import RPN_PQ_EE

import sys
import os

sys.path.append(os.path.join(os.getcwd(), "models"))


def get_model_weights(model, version):
    weights = getattr(M, model)
    return getattr(weights, version)


def ResNet(
    weights="ResNet152_Weights", version="IMAGENET1K_V1", progress=False, **kwargs
):
    weights = get_model_weights(weights, version)
    return M.resnet152(weights=weights, progress=progress, **kwargs)


def ViT(weights="ViT_B_32_Weights", version="IMAGENET1K_V1", progress=False, **kwargs):
    weights = get_model_weights(weights, version)
    return M.vit_b_32(weights=weights, progress=progress, **kwargs)


def Swin_V2(
    weights="Swin_V2_B_Weights", version="IMAGENET1K_V1", progress=False, **kwargs
):
    weights = get_model_weights(weights, version)
    return M.swin_v2_b(weights=weights, progress=progress, **kwargs)


__all__ = ["ResNet", "RPN", "RPN_P", "RPN_PQ", "RPN_PQ_EE", "ViT", "Swin_V2"]
