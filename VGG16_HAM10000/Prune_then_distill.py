from pruneutils import global_pruning_model
import torch
import torch.nn as nn
import torch.optim as optim
import os
from utils import evaluate_accuracy, get_data_loaders, device
from torchvision.models import vgg16, vgg11
from kd import train_student_with_distillation

train_loader, test_loader = get_data_loaders(batch_size=64)
alpha = 0.7
temperature = 15

# Set up the VGG16 teacher model
teacher = torch.load(os.path.join(
    "Original_Models", "main_model.pt"), map_location=device)

# Set up the optimizer and loss function for teacher training
optimizer_teacher = optim.SGD(
    teacher.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
criterion_teacher = nn.CrossEntropyLoss()


# Set up the smaller VGG16 student model
student = vgg11(num_classes=100)
student.classifier[6] = nn.Linear(4096, 100)
student = student.to(device)

# Set up the optimizer and loss function for student training
optimizer_student = optim.SGD(
    student.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
criterion_student = nn.KLDivLoss()


# Prune 30% Teacher
level = 0.3
teacher.train()
global_pruning_model(teacher, level, 0, train_loader, test_loader)
accuracy = evaluate_accuracy(teacher, test_loader)

torch.save(teacher, os.path.join("Teacher_student",
           f"Teacher_{level}_{accuracy:.2f}.pt"))

train_student_with_distillation(teacher, student, train_loader, test_loader,
                                optimizer_student, criterion_teacher, criterion_student, temperature, alpha, 100)

accuracy = evaluate_accuracy(student, test_loader)

torch.save(student, os.path.join("Teacher_student",
           f"Student_{level}_{accuracy:.2f}.pt"))


# Prune 50% Teacher
level = 0.5
teacher.train()
global_pruning_model(teacher, level, 0.3, train_loader, test_loader)
accuracy = evaluate_accuracy(teacher, test_loader)

torch.save(teacher, os.path.join("Teacher_student",
           f"Teacher_{level}_{accuracy:.2f}.pt"))

# Set up the smaller VGG16 student model
student = vgg11(num_classes=100)
student.classifier[6] = nn.Linear(4096, 100)
student = student.to(device)

# Set up the optimizer and loss function for student training
optimizer_student = optim.SGD(
    student.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
criterion_student = nn.KLDivLoss()

train_student_with_distillation(teacher, student, train_loader, test_loader,
                                optimizer_student, criterion_teacher, criterion_student, temperature, alpha, 100)

accuracy = evaluate_accuracy(student, test_loader)

torch.save(student, os.path.join("Teacher_student",
           f"Student_{level}_{accuracy:.2f}.pt"))


# Prune 80% Teacher
level = 0.8
teacher.train()
global_pruning_model(teacher, level, 0.5, train_loader, test_loader)
accuracy = evaluate_accuracy(teacher, test_loader)

torch.save(teacher, os.path.join("Teacher_student",
           f"Teacher_{level}_{accuracy:.2f}.pt"))

# Set up the smaller VGG16 student model
student = vgg11(num_classes=100)
student.classifier[6] = nn.Linear(4096, 100)
student = student.to(device)

# Set up the optimizer and loss function for student training
optimizer_student = optim.SGD(
    student.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
criterion_student = nn.KLDivLoss()

train_student_with_distillation(teacher, student, train_loader, test_loader,
                                optimizer_student, criterion_teacher, criterion_student, temperature, alpha, 100)

accuracy = evaluate_accuracy(student, test_loader)

torch.save(student, os.path.join("Teacher_student",
           f"Student_{level}_{accuracy:.2f}.pt"))


# Prune 90% Teacher
level = 0.9
teacher.train()
global_pruning_model(teacher, level, 0.8, train_loader, test_loader)
accuracy = evaluate_accuracy(teacher, test_loader)

torch.save(teacher, os.path.join("Teacher_student",
           f"Teacher_{level}_{accuracy:.2f}.pt"))

# Set up the smaller VGG16 student model
student = vgg11(num_classes=100)
student.classifier[6] = nn.Linear(4096, 100)
student = student.to(device)

# Set up the optimizer and loss function for student training
optimizer_student = optim.SGD(
    student.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
criterion_student = nn.KLDivLoss()

train_student_with_distillation(teacher, student, train_loader, test_loader,
                                optimizer_student, criterion_teacher, criterion_student, temperature, alpha, 100)

accuracy = evaluate_accuracy(student, test_loader)

torch.save(student, os.path.join("Teacher_student",
           f"Student_{level}_{accuracy:.2f}.pt"))
