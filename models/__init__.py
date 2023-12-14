from torchvision.models import resnet152
'''
from torchvision.models import (
    resnet152,
    ResNet152_Weights,
    vit_b_32,
    ViT_B_32_Weights,
    swin_v2_b,
    Swin_V2_B_Weights,
)
'''

import sys
import os

sys.path.append(os.path.join(os.getcwd(), "models"))


def ResNet(pretrained = True):
    return resnet152(weights=weights, pretrained=True)

'''
import ViT
from RPN import RPN
=======
def ViT(weights=ViT_B_32_Weights.IMAGENET1K_V1, progress=False, **kwargs):
    return vit_b_32(weights=weights, progress=progress, **kwargs)


def Swin_V2(weights=Swin_V2_B_Weights.IMAGENET1K_V1, progress=False, **kwargs):
    return swin_v2_b(weights=weights, progress=progress, **kwargs)

'''
import RPN

import RPN_P
import RPN_PQ
import RPN_PQ_EE

__all__ = ["ResNet", "RPN", "RPN_P", "RPN_PQ", "RPN_PQ_EE", "ViT", "Swin_V2"]
