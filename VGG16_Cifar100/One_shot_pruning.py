from utils import evaluate_accuracy, get_data_loaders ,device
from trains import train_teacher
import torch
import os
from pruneutils import global_pruning_model
import numpy as np



# set random seed
torch.manual_seed(42)
np.random.seed(42)

train_loader , test_loader = get_data_loaders()

# Prune 30%
model = torch.load(os.path.join("Original_Models",
                   "main_model.pt"), map_location=device)
pruning_level, accuracy = global_pruning_model(
    model, 0.3, 0, train_loader, test_loader)
torch.save(model, os.path.join("One_Shot_pruned",
           f"pruned_{pruning_level}_acc_{accuracy:.2f}.pt"))
del(model)

# Prune 50%
model = torch.load(os.path.join("Original_Models",
                   "main_model.pt"), map_location=device)
pruning_level, accuracy = global_pruning_model(
    model, 0.5, 0, train_loader, test_loader)
torch.save(model, os.path.join("One_Shot_pruned",
           f"pruned_{pruning_level}_acc_{accuracy:.2f}.pt"))
del(model)

# prune 80%
model = torch.load(os.path.join("Original_Models",
                   "main_model.pt"), map_location=device)
pruning_level, accuracy = global_pruning_model(
    model, 0.8, 0, train_loader, test_loader)
torch.save(model, os.path.join("One_Shot_pruned",
           f"pruned_{pruning_level}_acc_{accuracy:.2f}.pt"))
del(model)

# Prune 90%
model = torch.load(os.path.join("Original_Models",
                   "main_model.pt"), map_location=device)
pruning_level, accuracy = global_pruning_model(
    model, 0.9, 0, train_loader, test_loader)
torch.save(model, os.path.join("One_Shot_pruned",
           f"pruned_{pruning_level}_acc_{accuracy:.2f}.pt"))
del(model)
