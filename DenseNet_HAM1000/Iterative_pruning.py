from utils import evaluate_accuracy, get_data_loaders, device
from trains import train_teacher
import torch
import os
from pruneutils import global_pruning, global_pruning_model
import numpy as np
import torch.optim as optim
import torch.nn as nn


# set random seed
torch.manual_seed(42)
np.random.seed(42)

train_loader, test_loader = get_data_loaders(batch_size=32)

model = torch.load(os.path.join("Original_Models",
                   "main_model.pt"), map_location=device)
optimizer = optim.SGD(model.parameters(), lr=0.001,
                      momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()


pruning_level = 0.1
prev_level = 0
num_epochs = 10
train_losses = []
test_losses = []
test_accs = []
path = "Iterative_pruned"


for level in range(9):
    global_pruning_model(model, pruning_level, prev_level,
                         train_loader, test_loader)
    prev_level = pruning_level
    pruning_level = pruning_level + 0.1
    if(level == 2):
        accuracy = evaluate_accuracy(model, test_loader)
        new_path = os.path.join(path, f"pruned_0.3_acc_{accuracy:.2f}.pt")
        #pruned 30%
        torch.save(model, new_path)
    elif(level == 4):
        accuracy = evaluate_accuracy(model, test_loader)
        new_path = os.path.join(path, f"pruned_0.5_acc_{accuracy:.2f}.pt")
        #pruned 50%
        torch.save(model, new_path)
    elif(level == 7):
        accuracy = evaluate_accuracy(model, test_loader)
        new_path = os.path.join(path, f"pruned_0.8_acc_{accuracy:.2f}.pt")
        #pruned 80%
        torch.save(model, new_path)
    elif(level == 8):
        accuracy = evaluate_accuracy(model, test_loader)
        new_path = os.path.join(path, f"pruned_0.9_acc_{accuracy:.2f}.pt")
        #pruned 90%
        torch.save(model, new_path)
