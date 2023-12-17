import torch
import copy
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from utils import get_args
from datasets import Imagenet1KTrain
from transforms import get_transform
from benchmarking import get_performance
from torch.utils.data import DataLoader
from config import get_config
import models
import timm
from RPN.co_advise import rednet

device = "cuda" if torch.cuda.is_available() else 'cpu'

def calculate_sparsity(weight):
    return float(100.0 * torch.sum(torch.isclose(weight, torch.zeros(weight.shape).to(device))) / weight.nelement())

def print_per_layer_sparsity(model):
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
            sparsity = calculate_sparsity(module.weight)
            print(f"Layer: {name}, Sparsity: {sparsity:.2f}%")


def print_global_sparsity(model):
    sparsity = 0.0
    num_weights = 0.0
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
            sparsity += float(torch.sum(torch.isclose(module.weight, torch.zeros(module.weight.shape).to(device))))
            num_weights += module.weight.nelement()
    
    global_sparsity = float(100.0 * (sparsity / num_weights))
    print(f"Global Sparsity: {global_sparsity:.2f}")

def get_per_layer_stats(model, quantiles):
    stats = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weights') and isinstance(module.weights, torch.nn.Parameter):

            weights = module.weights.data.view(-1)
            abs_weights = torch.abs(weights)

            layer_stats = {
                'mean': torch.mean(abs_weights).item(),
                'std': torch.std(abs_weights).item(),
                'median': torch.median(abs_weights).item(),
                'mode': torch.mode(abs_weights).item(),
                'min': torch.min(abs_weights).item(),
                'max': torch.max(abs_weights).item()
            }

            for q in quantiles:
                layer_stats[f'quantile_{q}']= torch.quantile(abs_weights, q).item()

            stats[name] = layer_stats
        
    return stats

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            if (i >= 2):
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss

def sensitivity_analysis(model, data_loader, criterion, threshold=0.01):
    original_model = copy.deepcopy(model).to(device)
    original_loss = evaluate_model(original_model, data_loader, criterion)
    sensitivity = {}

    for name, module in original_model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):

            for i in range(module.weight.data.size(0)):
                
                temp = module.weight.data[i].clone()
                module.weight.data[i] = 0.

                pruned_loss = evaluate_model(original_model, data_loader, criterion)
                
                module.weight.data[i] = temp
                sensitivity[(name, i)] = pruned_loss - original_loss
                print(f"Module: {name}, Parameter:[{i}], pruned_loss = {pruned_loss:.6f}, sensitivity:{sensitivity[(name,i)]:.6f}")

    return sensitivity

class Magnitude_BasedPruning(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def __init__(self, amount):
        self.amount = amount
    
    def compute_mask(self, t, default_mask):
        
        num_elements = t.numel()
        num_pruned = int(self.amount * num_elements)
        pruned_amount = min(num_pruned, (num_elements-1))
        threshold= torch.topk(torch.abs(t).view(-1), pruned_amount, largest=False).values[-1]
        mask = torch.abs(t).ge(threshold).type(default_mask.dtype)

        return mask

def magnitude_prune(module, name, amount):
    pruner = Magnitude_BasedPruning(amount)
    pruner.apply(module, name, amount)
    return module

def apply_magnitude_pruning(model, amount):
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and isinstance(module, nn.Linear):
            magnitude_prune(module, 'weight', amount)
        elif hasattr(module, 'weight') and isinstance(module, nn.Conv2d):
            magnitude_prune(module, 'weight', amount)

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

def save_model(model):
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            prune.remove(module, "weight")
    
    print("%%===============Saving the Pruned Model==============>>>>>>>>>%%")
    torch.save(model.state_dict(), "/vast/km5939/ECEGY-9143_High-Performance-Machine-Learning_Capstone-Project/models/RPN_P/pruned_model_2.pth")
    print("saved Model: pruned_model_2.pth")
    print("===============================================================")

def calculate_pruned_parameters(model_s):
    pruned = 0
    unpruned = 0
    for name, module in model_s.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            if hasattr(module, 'weight_mask'):
                
                mask = module.weight_mask
                unp = torch.sum(mask).item()
                unpruned += unp
                pruned += mask.numel() - unp
    return pruned, unpruned

def main():
    args = get_args()
    cfg = get_config()

    train_epochs = 1

    model = getattr(models, args.model)()
    transform = get_transform(args.model, "Imagenet1KTrain")
    dataset = Imagenet1KTrain("/vast/km5939/data", "train", transform=transform)

    print("=" + "*=" * 18)
    print("pre pruning sparsity stats: \n")
    #print_per_layer_sparsity(model)
    print_global_sparsity(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters pre pruning: {total_params}")
    pruned_model = copy.deepcopy(model).to(device)
    dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            shuffle=True,
        )
    criterion = torch.nn.CrossEntropyLoss()

    #sensitivity_map = sensitivity_analysis(model, dataloader, criterion)
    #print("/**" * 10)
    #print("sensitivity_map:-\n")
    #print(sensitivity_map)
    #print("/**" * 10)
    stats = get_per_layer_stats(pruned_model, [0.20, 0.25, 0.5, 0.70, 0.75])
    optimizer = torch.optim.AdamW(pruned_model.parameters(), lr = 0.0001)
    apply_magnitude_pruning(pruned_model, 0.25)
    
    print("**" + "/**" * 10) 
    print("post pruning sparsity stats: \n")
    #print_per_layer_sparsity(pruned_model)
    print_global_sparsity(pruned_model)
    total_params_1 = sum(p.numel() for p in pruned_model.parameters())
    print(f"Total parameters post pruning: {total_params_1}")
    pruned_params, un_pruned_params = calculate_pruned_parameters(pruned_model)
    print(f"pruned parameters: {pruned_params}")
    print(f"unpruned_parameters: {un_pruned_params}")
    
    dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            shuffle=True,
        )
    for n in range(train_epochs):
        loss, accuracy = train_one_epoch(pruned_model, dataloader, optimizer, criterion, device)
        print(f'Epoch: {n+1}/{train_epochs}, loss = {loss}, Accuracy = {accuracy}')
    
    print("**" + "/**" * 10)
    print("=" + "*=" * 18)

    save_model(pruned_model)

if __name__ == '__main__':
    main()

