from trains import train_teacher
from utils import get_data_loaders,setup_teacher_model, setup_student_model, device
import torch.nn as nn
import torch.optim as optim
import torch 
import os
import numpy as np
from torchvision.models import densenet161
# set random seed
torch.manual_seed(42)
np.random.seed(42)


train_loader, test_loader = get_data_loaders(batch_size=32)

# Train the main Model before pruning.
teacher, optimizer_teacher, criterion_teacher = setup_teacher_model()
optimizer_teacher = optim.SGD(
    teacher.parameters(), momentum=0.9,  lr=0.001, weight_decay=5e-4)
train_teacher(teacher, optimizer_teacher, criterion_teacher,
              train_loader, test_loader, 100)
torch.save(teacher, os.path.join("Original_Models", "main_model.pt"))


# Train a student Model 
student, _, _ = setup_student_model()
student = densenet161(num_classes=7)
student.classifier = nn.Linear(2208, 7)
student = student.to(device)

optimizer_student = optim.SGD(
    student.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
criterion_student = nn.CrossEntropyLoss()
train_teacher(student, optimizer_student, criterion_student,
              train_loader, test_loader, 100)

torch.save(student, os.path.join("Original_Models", "student_model.pt"))

