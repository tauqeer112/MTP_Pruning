from utils import evaluate_accuracy, get_data_loaders, device
from trains import train_teacher
import torch
import os
import torch.optim as optim
import torch.nn as nn
from pruneutils import train
import copy
import torch.nn.utils.prune as prune
from utils import evaluate_accuracy
import numpy as np

# set random seed
torch.manual_seed(42)
np.random.seed(42)


# prunable layers and blocks

prunable_layers = ['features.conv0', 'features.denseblock1', 'features.transition1', 'features.denseblock2',
                   'features.transition2', 'features.denseblock3', 'features.transition3', 'features.denseblock4', 'classifier']

train_loader, test_loader = get_data_loaders(batch_size=32)


epochs = 15
# Pruning each layers to 30%
model = torch.load(os.path.join("Original_Models",
                   "main_model.pt"), map_location=device)
optimizer = optim.SGD(model.parameters(), lr=0.001,
                      momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
amount = 0.3  # pruning amount

for layer in prunable_layers:
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm2d):
            if name.startswith(layer):
                parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    train_teacher(model, optimizer, criterion,
                  train_loader, test_loader, epochs)
    # Count the number of parameters and non-zero parameters in the model
    total_params = 0
    nonzero_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm2d):
            total_params += module.weight.nelement()
            nonzero_params += torch.sum(module.weight != 0)

    # Compute the sparsity of the model
    sparsity = 1 - (nonzero_params / total_params)
    print(f"Sparsity: {sparsity:.2%}")

    accuracy = evaluate_accuracy(model, test_loader)
    path = os.path.join("Layerwise_pruned", '30')
    torch.save(model, os.path.join(
        path, f"{layer}_sparsity_{sparsity:.4%}_acc_{accuracy:.2f}.pt"))

del(model)


# pruning each layer to 50%
model = torch.load(os.path.join("Original_Models",
                   "main_model.pt"), map_location=device)
optimizer = optim.SGD(model.parameters(), lr=0.001,
                      momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
amount = 0.5

for layer in prunable_layers:
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm2d):
            if name.startswith(layer):
                parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    train_teacher(model, optimizer, criterion,
                  train_loader, test_loader, epochs)
    # Count the number of parameters and non-zero parameters in the model
    total_params = 0
    nonzero_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm2d):
            total_params += module.weight.nelement()
            nonzero_params += torch.sum(module.weight != 0)

    # Compute the sparsity of the model
    sparsity = 1 - (nonzero_params / total_params)
    print(f"Sparsity: {sparsity:.2%}")

    accuracy = evaluate_accuracy(model, test_loader)
    path = os.path.join("Layerwise_pruned", '50')
    torch.save(model, os.path.join(
        path, f"{layer}_sparsity_{sparsity:.4%}_acc_{accuracy:.2f}.pt"))

del(model)

# pruning each layer to 80%
model = torch.load(os.path.join("Original_Models",
                   "main_model.pt"), map_location=device)
optimizer = optim.SGD(model.parameters(), lr=0.001,
                      momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
amount = 0.8

for layer in prunable_layers:
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm2d):
            if name.startswith(layer):
                parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    train_teacher(model, optimizer, criterion,
                  train_loader, test_loader, epochs)
    # Count the number of parameters and non-zero parameters in the model
    total_params = 0
    nonzero_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm2d):
            total_params += module.weight.nelement()
            nonzero_params += torch.sum(module.weight != 0)

    # Compute the sparsity of the model
    sparsity = 1 - (nonzero_params / total_params)
    print(f"Sparsity: {sparsity:.2%}")

    accuracy = evaluate_accuracy(model, test_loader)
    path = os.path.join("Layerwise_pruned", '80')
    torch.save(model, os.path.join(
        path, f"{layer}_sparsity_{sparsity:.4%}_acc_{accuracy:.2f}.pt"))

del(model)

# pruning each layer to 90%
model = torch.load(os.path.join("Original_Models",
                   "main_model.pt"), map_location=device)
optimizer = optim.SGD(model.parameters(), lr=0.001,
                      momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
amount = 0.9

for layer in prunable_layers:
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm2d):
            if name.startswith(layer):
                parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    train_teacher(model, optimizer, criterion,
                  train_loader, test_loader, epochs)
    # Count the number of parameters and non-zero parameters in the model
    total_params = 0
    nonzero_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm2d):
            total_params += module.weight.nelement()
            nonzero_params += torch.sum(module.weight != 0)

    # Compute the sparsity of the model
    sparsity = 1 - (nonzero_params / total_params)
    print(f"Sparsity: {sparsity:.2%}")

    accuracy = evaluate_accuracy(model, test_loader)
    path = os.path.join("Layerwise_pruned", '90')
    torch.save(model, os.path.join(
        path, f"{layer}_sparsity_{sparsity:.4%}_acc_{accuracy:.2f}.pt"))
