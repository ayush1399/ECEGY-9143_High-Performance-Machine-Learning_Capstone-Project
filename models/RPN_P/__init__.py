import torch
from RPN.co_advise import models

def RPN_P():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pruned_model = models.deit_small_distilled_patch16_224().to(device)
    pruned_model.load_state_dict(torch.load("/vast/km5939/ECEGY-9143_High-Performance-Machine-Learning_Capstone-Project/models/RPN_P/pruned_model.pth", map_location=device))
    return pruned_model

__all__ = ["RPN_P"]
