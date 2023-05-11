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

prunable_layers = ['features.0', 'features.2', 'features.5', 'features.7', 'features.10', 'features.12', 'features.14',
                   'features.17', 'features.19', 'features.21', 'features.24', 'features.26', 'features.28', 'classifier.6', 'classifier.3', 'classifier.0']

train_loader, test_loader = get_data_loaders(batch_size=64)
epochs = 100


# Prune 30%
model = torch.load(os.path.join("Original_Models",
                   "main_model.pt"), map_location=device)

accuracy = evaluate_accuracy(model, test_loader)
print(f'Test Accuracy: {accuracy:.2f}%')


optimizer = optim.SGD(model.parameters(), lr=0.001,
                      momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

amount = 0.3

for name, module in model.named_modules():
    if name in prunable_layers:  # or isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=amount)
        train_teacher(model, optimizer, criterion,
                      train_loader, test_loader, epochs)

        accuracy = evaluate_accuracy(model, test_loader)
        # Count the number of parameters and non-zero parameters in the model
        total_params = 0
        nonzero_params = 0
        for x, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                total_params += module.weight.nelement()
                nonzero_params += torch.sum(module.weight != 0)

        # Compute the sparsity of the model
        sparsity = 1 - (nonzero_params / total_params)
        print(f"Sparsity: {sparsity:.2%}")
        path = os.path.join("Layerwise_pruned", '30')
        torch.save(model, os.path.join(
            path, f"{name}_sparsity_{sparsity:.3%}_acc_{accuracy:.2f}.pt"))


# Prune 50%
model = torch.load(os.path.join("Original_Models",
                   "main_model.pt"), map_location=device)

accuracy = evaluate_accuracy(model, test_loader)
print(f'Test Accuracy: {accuracy:.2f}%')


optimizer = optim.SGD(model.parameters(), lr=0.001,
                      momentum=0.9, weight_decay=5e-4)

criterion = nn.CrossEntropyLoss()
amount = 0.5

for name, module in model.named_modules():
    if name in prunable_layers:  # or isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=amount)
        train_teacher(model, optimizer, criterion,
                      train_loader, test_loader, epochs)

        accuracy = evaluate_accuracy(model, test_loader)
        # Count the number of parameters and non-zero parameters in the model
        total_params = 0
        nonzero_params = 0
        for x, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                total_params += module.weight.nelement()
                nonzero_params += torch.sum(module.weight != 0)

        # Compute the sparsity of the model
        sparsity = 1 - (nonzero_params / total_params)
        print(f"Sparsity: {sparsity:.2%}")
        path = os.path.join("Layerwise_pruned", '50')
        torch.save(model, os.path.join(
            path, f"{name}_sparsity_{sparsity:.3%}_acc_{accuracy:.2f}.pt"))


# Prune 80%
model = torch.load(os.path.join("Original_Models",
                   "main_model.pt"), map_location=device)

accuracy = evaluate_accuracy(model, test_loader)
print(f'Test Accuracy: {accuracy:.2f}%')


optimizer = optim.SGD(model.parameters(), lr=0.001,
                      momentum=0.9, weight_decay=5e-4)

criterion = nn.CrossEntropyLoss()
amount = 0.8

for name, module in model.named_modules():
    if name in prunable_layers:  # or isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=amount)
        train_teacher(model, optimizer, criterion,
                      train_loader, test_loader, epochs)
        accuracy = evaluate_accuracy(model, test_loader)
        # Count the number of parameters and non-zero parameters in the model
        total_params = 0
        nonzero_params = 0
        for x, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                total_params += module.weight.nelement()
                nonzero_params += torch.sum(module.weight != 0)

        # Compute the sparsity of the model
        sparsity = 1 - (nonzero_params / total_params)
        print(f"Sparsity: {sparsity:.2%}")
        path = os.path.join("Layerwise_pruned", '80')
        torch.save(model, os.path.join(
            path, f"{name}_sparsity_{sparsity:.3%}_acc_{accuracy:.2f}.pt"))


# Prune 90%
model = torch.load(os.path.join("Original_Models",
                   "main_model.pt"), map_location=device)

accuracy = evaluate_accuracy(model, test_loader)
print(f'Test Accuracy: {accuracy:.2f}%')


optimizer = optim.SGD(model.parameters(), lr=0.001,
                      momentum=0.9, weight_decay=5e-4)

criterion = nn.CrossEntropyLoss()
amount = 0.9

for name, module in model.named_modules():
    if name in prunable_layers:  # or isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=amount)
        train_teacher(model, optimizer, criterion,
                      train_loader, test_loader, epochs)

        accuracy = evaluate_accuracy(model, test_loader)
        # Count the number of parameters and non-zero parameters in the model
        total_params = 0
        nonzero_params = 0
        for x, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                total_params += module.weight.nelement()
                nonzero_params += torch.sum(module.weight != 0)

        # Compute the sparsity of the model
        sparsity = 1 - (nonzero_params / total_params)
        print(f"Sparsity: {sparsity:.2%}")
        path = os.path.join("Layerwise_pruned", '90')
        torch.save(model, os.path.join(
            path, f"{name}_sparsity_{sparsity:.3%}_acc_{accuracy:.2f}.pt"))
