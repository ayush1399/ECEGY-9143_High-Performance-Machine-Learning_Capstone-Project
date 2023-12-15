import torch
from RPN.co_advise import models

def RPN():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load('/vast/km5939/ECEGY-9143_High-Performance-Machine-Learning_Capstone-Project/models/RPN/co_advise/ckpt.pth', map_location= device)
    my_trained_model =  models.deit_small_distilled_patch16_224().to(device)
    my_trained_model.load_state_dict(checkpoint['model'])
    return my_trained_model

__all__ = ["RPN"]
