import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import vgg16 ,vgg11, densenet169 ,densenet161
import numpy as np
import torchvision
import torchvision.models as models
import copy
import os

device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
# device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

# set random seed
torch.manual_seed(42)
np.random.seed(42)

import pandas as pd
from torch.utils.data import Dataset 
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms



def get_data_loaders(batch_size=128, num_workers=32):
# Define transforms
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load the CIFAR-10 training and test datasets
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    # Set up the data loaders for the CIFAR-10 datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader



def setup_teacher_model():
    # Set up the VGG16 teacher model
    teacher = densenet169(pretrained=True)
    teacher.classifier = nn.Linear(1664, 100)
    # teacher.classifier[6] = nn.Linear(4096, 7)
    teacher = teacher.to(device)

    # Set up the optimizer and loss function for teacher training
    optimizer_teacher = optim.SGD(teacher.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4) 
    criterion_teacher = nn.CrossEntropyLoss()

    return teacher, optimizer_teacher, criterion_teacher


# def setup_student_model():
#     # Set up the smaller VGG16 student model
#     student = vgg16(num_classes=10)
#     student.classifier[6] = nn.Linear(4096, 10)
#     student = student.to(device)

#     # Set up the optimizer and loss function for student training
#     optimizer_student = optim.SGD(student.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
#     criterion_student = nn.KLDivLoss()

#     return student, optimizer_student, criterion_student

def setup_student_model():
    # Set up the smaller VGG16 student model
    student = densenet161(pretrained=True)
    student.classifier = nn.Linear(2208, 100)
    student = student.to(device)

    optimizer_student = optim.SGD(student.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4) 
    criterion_student = nn.KLDivLoss()

    return student, optimizer_student, criterion_student


def evaluate_accuracy(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total * 100
    return accuracy


def sparsity(model):
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
    return sparsity