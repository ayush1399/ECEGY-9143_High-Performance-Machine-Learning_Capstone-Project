from models import RPN_P

import torch.nn.functional as F
import torch.nn as nn
import torch


class EarlyExitResNet(nn.Module):
    def __init__(self, pretrained_model, exit_layer):
        super(EarlyExitResNet, self).__init__()
        self.pretrained_model = pretrained_model
        self.exit_layer = exit_layer

    def forward(self, x):
        # Forward pass up to the early exit layer
        for i, layer in enumerate(self.pretrained_model.children()):
            x = layer(x)
            if i == self.exit_layer:
                exit_output = x
                break

        # Compute entropy
        exit_output = F.softmax(exit_output, dim=1)
        entropy = -(exit_output * torch.log(exit_output)).sum(dim=1).mean()

        return exit_output, entropy


def RPN_EE(exit_layer=40, *args, **kwargs):
    pretrained_model = RPN_P(*args, **kwargs)
    return EarlyExitResNet(pretrained_model, exit_layer=exit_layer)
