import torch
from RPN.co_advise import models


def RPN_P(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pruned_model = models.deit_small_distilled_patch16_224().to(device)
    pruned_model.load_state_dict(torch.load(path, map_location=device))
    return pruned_model


__all__ = ["RPN_P"]
