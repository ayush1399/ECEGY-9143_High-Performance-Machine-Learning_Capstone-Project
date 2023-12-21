from models import RPN_EE, RPN
from datasets import Imagenet1KVal
from torch.utils.data import DataLoader

import numpy as np
import torch


def analyze(exit_layer=40):
    model = RPN_EE(exit_layer=exit_layer)
    dataset = Imagenet1KVal("./")
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )
    model.eval()
    model.cuda()

    ee_outputs, ee_entropy = [], []
    outputs, labels = [], []

    for _, (x, y) in enumerate(dataloader):
        x = x.cuda()
        y = y.cuda()
        output, e = model(x)
        ee_outputs.append(output.detach().cpu())
        ee_entropy.append(e.detach().cpu())

    ee_outputs = torch.cat(outputs, dim=0)
    ee_entropy = torch.cat(ee_entropy, dim=0)

    model = RPN()
    model.eval()
    model.cuda()

    for _, (x, y) in enumerate(dataloader):
        x = x.cuda()
        y = y.cuda()
        output = model(x)
        outputs.append(output.detach().cpu())
        labels.append(y.detach().cpu())

    outputs = torch.cat(outputs, dim=0)
    labels = torch.cat(labels, dim=0)

    acc = []

    for entropy_t in [0.33, 0.5, 0.66]:
        ee_flag = torch.zeros_like(ee_entropy)
        ee_flag[ee_entropy < entropy_t] = 1

        preds = torch.zeros_like(outputs)
        preds[ee_flag] = ee_outputs[ee_flag]
        preds[1 - ee_flag] = outputs[1 - ee_flag]

        acc.append((preds.argmax(dim=1) == labels).float().mean().item())

    return acc


if __name__ == "__main__":
    for exit_layer in [40, 36, 32, 28, 24, 20, 16, 12, 8, 4]:
        acc = analyze(exit_layer=exit_layer)
        print(f"*" * 20)
        print(f"EE Exit Layer: {exit_layer}")
        print(f"EE Threshold: 0.33, Acc: {acc[0]}")
        print(f"EE Threshold: 0.50, Acc: {acc[1]}")
        print(f"EE Threshold: 0.66, Acc: {acc[2]}")
        print(f"*" * 20)
