import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.models as models
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import copy
import os
from utils import evaluate_accuracy, get_data_loaders, device
from trains import train_teacher
import gc


# set random seed
torch.manual_seed(42)
np.random.seed(42)

train_loader, test_loader = get_data_loaders(batch_size=64)


def global_pruning_model(model, pruning_level, prev_level, train_loader, test_loader):
    if prev_level != 0:
        pruning_level = 1 - ((1-pruning_level)/(1-prev_level))

    parameters_to_prune = [
        (model.features[i], 'weight') for i in range(len(model.features))
        if isinstance(model.features[i], nn.Conv2d)
    ] + [
        (model.classifier[i], 'weight') for i in range(len(model.classifier))
        if isinstance(model.classifier[i], nn.Linear)
    ]
    print(f"Applying global pruning with sparsity level {pruning_level}")

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_level)

    # Count the number of parameters and non-zero parameters in the model
    total_params = 0
    nonzero_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            total_params += module.weight.nelement()
            nonzero_params += torch.sum(module.weight != 0)

    # Compute the sparsity of the model
    sparsity = 1 - (nonzero_params / total_params)
    print(f"Sparsity: {sparsity:.2%}")

    optimizer = optim.SGD(model.parameters(), lr=0.001,
                          momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    # train_teacher(teacher , optimizer_teacher , criterion_teacher , train_loader , test_loader , 40)
    train_teacher(model, optimizer, criterion, train_loader, test_loader, 10)

    accuracy = evaluate_accuracy(model, test_loader)

    print(f'{pruning_level * 100}%, Test Accuracy: {accuracy:.2f}%')

    # Count the number of parameters and non-zero parameters in the model
    total_params = 0
    nonzero_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            total_params += module.weight.nelement()
            nonzero_params += torch.sum(module.weight != 0)

    # Compute the sparsity of the model
    sparsity = 1 - (nonzero_params / total_params)
    print(f"Sparsity: {sparsity:.2%}")
    return model
