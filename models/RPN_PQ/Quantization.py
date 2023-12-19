import copy
import random
import os
import models
import timm
import torch
import torch.nn as nn
from utils import get_args
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.quantization import  QConfig, prepare, convert, MinMaxObserver, PerChannelMinMaxObserver
from utils import get_args
from datasets import Imagenet1KTrain
from transforms import get_transform
from config import get_config
from RPN.co_advise import rednet

class QuantizedModel(torch.nn.Module):
    def __init__(self, model):
        super(QuantizedModel, self).__init__()
        self.model_fp32 = model
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x

class CustomWeightObserver(PerChannelMinMaxObserver):
    def __init__(self, 
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric, 
                ch_axis=0, 
                **kwargs):
        
        super(CustomWeightObserver, self).__init__(dtype=dtype, qscheme=qscheme, ch_axis = ch_axis, **kwargs)

class CustomActivationObserver(MinMaxObserver):
    def __init__(self,
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
                **kwargs):
        
        super(CustomActivationObserver, self).__init__(dtype=dtype, qscheme=qscheme, **kwargs)

class CustomQConfig:
    def __init__(self):
        self.conv_qconfig = QConfig(
            activation=CustomActivationObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
            weight=CustomWeightObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0)
        )

        # QConfig for linear layers
        self.linear_qconfig = QConfig(
            activation=CustomActivationObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
            weight=CustomWeightObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0)
        )

def apply_custom_quantization(model, custom_qconfig):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            apply_custom_quantization(module, custom_qconfig)
        if isinstance(module, nn.Linear):
            module.qconfig = custom_qconfig.linear_qconfig
        elif isinstance(module, nn.Conv2d):
            module.qconfig = custom_qconfig.conv_qconfig

def calibrate_model(model, calibration_loader, device):
    model.eval()
    with torch.no_grad():
        for batch, _ in calibration_loader:
            batch = batch.to(device)
            model(batch)


def Calibration_data_loader(dataset):

    indices = []
    for class_index in range(len(dataset.classes)):
        cls_indices = [i for i, (_,label) in enumerate(dataset.samples) if label == class_index]
        subset_indices = random.sample(cls_indices, min(10, len(cls_indices)))
        indices.extend(subset_indices)
    
    calibration_data = Subset(dataset, indices)

    calibration_loader = DataLoader(calibration_data, batch_size=32, shuffle=False)
    return calibration_loader

def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("Model size: %.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')

def save_model(mdl):
    torch.save(mdl.state_dict(), "/vast/km5939/ECEGY-9143_High-Performance-Machine-Learning_Capstone-Project/models/RPN_PQ/quantized_model.pth")

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    T = 1
    conv_teacher = timm.create_model("regnety_040")
    conv_teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(conv_teacher)
    conv_teacher.load_state_dict(torch.load("/vast/km5939/ECEGY-9143_High-Performance-Machine-Learning_Capstone-Project/models/RPN/co_advise/conv_teacher.pth", map_location="cpu")["model"])
    conv_teacher.to(device)

    inv_teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(rednet.Red101())
    inv_teacher.load_state_dict(torch.load("/vast/km5939/ECEGY-9143_High-Performance-Machine-Learning_Capstone-Project/models/RPN/co_advise/inv_teacher.pth", map_location="cpu")["model"])
    inv_teacher.to(device)

    print("Training: Loaded teacher models")
    conv_teacher.eval()
    inv_teacher.eval()
    running_loss = 0.0
    accuracy = 0.0
    total_predictions = 0
    correct_predictions = 0
    kl = nn.KLDivLoss(reduction='batchmean')

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        with torch.no_grad():
            y_conv = conv_teacher(inputs)
            y_inv = inv_teacher(inputs)
        outputs, y_kd_conv, y_kd_inv = model(inputs)

        ce_loss = criterion(outputs, targets)
        conv_kd_loss = kl(F.log_softmax(y_kd_conv / T, dim=-1), F.softmax(y_conv / T, dim=-1)) * T * T
        inv_kd_loss = kl(F.log_softmax(y_kd_inv / T, dim=-1), F.softmax(y_inv / T, dim=-1)) * T * T
        loss = ce_loss+conv_kd_loss+inv_kd_loss

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predictions = torch.max(outputs, 1)
        total_predictions += targets.size(0)
        correct_predictions += (predictions == targets).sum().item()

    average_loss = running_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    return average_loss, accuracy

def main():
    args = get_args()
    custom_qconfig = CustomQConfig()
    cfgs = get_config()

    train_epochs = 1
    model = getattr(models, args.model)()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transform(args.model, "Imagenet1KTrain")
    dataset = Imagenet1KTrain("/vast/km5939/data", "train", transform=transform)
    model_bef_q = copy.deepcopy(model).to(device)
    
    print("=" + "*=" * 18)
    print()
    print("Model Size before quantization: ")
    print_model_size(model_bef_q)
    print()
    apply_custom_quantization(model_bef_q, custom_qconfig)

    model_prepared = prepare(model_bef_q, inplace=True)
    
    print("**" + "/**" * 12)
    print("========Calibrating the model==========")
    print()
    calibration_loader = Calibration_data_loader(dataset)
    calibrate_model(model_prepared, calibration_loader, device)

    print("========Finished Calibration===========")
    print()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model_prepared.parameters(), lr = 1e-04)

    dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            shuffle=True,
        )

    print(f"Finetuning the model on epochs = {train_epochs}")
    print()
    for n in range(train_epochs):
        loss, accuracy = train_one_epoch(model_prepared, dataloader, optimizer, criterion, device)
        print(f'Epoch: {n+1}/{train_epochs}, loss = {loss}, Accuracy_top1 = {accuracy}')

    print()
    print("Finished finetuning the model")
    print("**" + "/**" * 12)

    model_prepared.to(torch.device("cpu"))
    quantized_model = convert(model_prepared, inplace=True)

    print("Model Size after quantization: ")
    print_model_size(quantized_model)
    quantized_model.to("cuda")
    print("%%========================Saving the quantized model======>>>>>>>>>>>>>>>>>>")
    save_model(quantized_model)
    print()
    print("%%=================Quantized Model: quantized_model.pth saved====>>>>>>>>>>>>")
    print()
    print("=" + "*=" * 18)
    
if __name__ == "__main__":
    main()
