import torch
import copy
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as f
from models.RPN.co_advise import models
from utils import get_args, get_dataset, pretty_print_perf
from benchmarking import get_performance
from torch.utils.data import DataLoader
from config import get_config

def calculate_sparsity(weight):
    return float(torch.sum(torch.isclose(weight, torch.zeros(weight.shape))) / weight.numel())

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
            sparsity += float(torch.sum(module.weight == 0))
            num_weights += module.numel()
    
    global_sparsity = float(sparsity / num_weights)
    print(f"Global Sparsity: {global_sparsity:.2f}")

def get_per_layer_stats(model, quantiles):
    stats = {}
    for name, module in model.named.modules():
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
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss

def sensitivity_analysis(model, data_loader, criterion, threshold=0.01):
    original_model = copy.deepcopy(model)
    original_loss = evaluate_model(model, data_loader, criterion)
    sensitivity = {}

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):  # or any other layer type
            original_weights = module.weight.data.clone()
            for i in range(module.weight.data.size(0)):
                
                module.weight.data[i] = 0.
                
                pruned_loss = evaluate_model(model, data_loader, criterion)
                
                module.weight.data = original_weights.clone()
                
                sensitivity[(name, i)] = pruned_loss - original_loss

    model.load_state_dict(original_model.state_dict())
    return sensitivity

class MagnitudeBasedPruning(prune.BasePruningMethod):

    PRUNING_TYPE = 'unstructured'

    def __init__(self, amount):
        super(MagnitudeBasedPruning, self).__init__()
        self.amount = amount
    
    def compute_mask(self, t, default_mask):
        
        mask = torch.abs(t).ge(self.amount).type(default_mask.dtype)
        return mask

def magnitude_prune(module, name, threshold):
    pruner = MagnitudeBasedPruning(threshold)
    pruner.apply(module, name)
    return module

def apply_magnitude_pruning(model, stats):
    for name, module in model.named_modules():
        if name in stats and hasattr(module, 'weight'):
            if isinstance(module, nn.Linear):
                magnitude_prune(module, 'weight', stats[name]['quantile_0.20'])
            elif isinstance(module, nn.Conv2d):
                magnitude_prune(module, 'weight', stats[name]['median'])

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    accuracy = 0.0
    total_predictions = 0
    correct_predictions = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets)

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
    cfg = get_config()

    train_epochs = 5

    model = getattr(models, args.model)()
    dataset = get_dataset(args, cfg)
    model = models.deit_small_distilled_patch16_224()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load('/model/path/ckpt.pth', map_location= device)
    model.load_state_dict(checkpoint['state_dict'])

    print("pre pruning sparsity stats: \n")
    print_per_layer_sparsity(model)
    print_global_sparsity(model)

    pruned_model = copy.deepcopy(model)
    dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            shuffle=False,
        )
    criterion = torch.nn.CrossEntropyLoss()

    sensitivity_map = sensitivity_analysis(pruned_model, dataloader, criterion)

    stats = get_per_layer_stats(pruned_model, [0.20, 0.25, 0.5, 0.70, 0.75])
    optimizer = torch.optim.Adam(pruned_model.parameters(), lr = 0.001)
    apply_magnitude_pruning(pruned_model, stats)
    
    for n in range(train_epochs):
        loss, accuracy = train_one_epoch(pruned_model, dataloader, optimizer, criterion, device)
        print(f'Epoch: {n}/{train_epochs}, loss = {loss}, Accuracy = {accuracy}')
        print_per_layer_sparsity(pruned_model)
        print_global_sparsity(pruned_model)

if __name__ == 'main':
    main()



