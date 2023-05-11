from trains import train_teacher
from utils import get_data_loaders, setup_teacher_model, setup_student_model
import torch.nn as nn
import torch.optim as optim
import torch
import os
import numpy as np

# set random seed
torch.manual_seed(42)
np.random.seed(42)


train_loader, test_loader = get_data_loaders(batch_size=128)
teacher, optimizer_teacher, criterion_teacher = setup_teacher_model()
optimizer_teacher = optim.SGD(
    teacher.parameters(), momentum=0.9,  lr=0.001, weight_decay=5e-4)
train_teacher(teacher, optimizer_teacher, criterion_teacher,
              train_loader, test_loader, 100)
torch.save(teacher, os.path.join("Original_Models", "main_model.pt"))


student, _, _ = setup_student_model()
optimizer_student = optim.SGD(
    student.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
criterion_student = nn.CrossEntropyLoss()
train_teacher(student, optimizer_student, criterion_student,
              train_loader, test_loader, 200)
torch.save(student, os.path.join("Original_Models", "student_model.pt"))
