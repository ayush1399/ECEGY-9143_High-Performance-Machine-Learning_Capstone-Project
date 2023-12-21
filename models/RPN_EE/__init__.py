from models import RPN_P

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


"/vast/km5939/ECEGY-9143_High-Performance-Machine-Learning_Capstone-Project/models/RPN_P/pruned_model.pth"
