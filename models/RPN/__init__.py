from RPN.co-advise import models

def RPN():
    return models.deit_small_distilled_patch16_224()

__all__ = ["RPN"]