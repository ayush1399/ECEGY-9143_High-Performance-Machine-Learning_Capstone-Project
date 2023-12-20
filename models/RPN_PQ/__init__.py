import torch
from RPN.co_advise import models
from .Quantization import CustomWeightObserver, CustomActivationObserver, CustomQConfig, apply_custom_quantization, QuantizedModel

def RPN_PQ():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_model = models.deit_small_distilled_patch16_224()
    q_config = CustomQConfig()
    apply_custom_quantization(q_model, q_config)
    model_prepared = torch.quantization.prepare(q_model, inplace=True)
    model_prepared.to("cpu")
    qu_model = torch.quantization.convert(model_prepared, inplace=True)
    qu_model.load_state_dict(torch.load("/vast/km5939/ECEGY-9143_High-Performance-Machine-Learning_Capstone-Project/models/RPN_PQ/quantized_model.pth", map_location="cpu"))
    quant_model = QuantizedModel(qu_model)
    quant_model.to(device)
    return quant_model

__all__ = ["RPN_PQ"]
