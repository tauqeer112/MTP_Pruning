from trains import train_teacher
from utils import get_data_loaders, setup_teacher_model, setup_student_model
import torch.nn as nn
import torch.optim as optim
import torch
import os
from Knowledge_Distillation import train_student_with_distillation
import numpy as np
from utils import device
# set random seed
torch.manual_seed(42)
np.random.seed(42)

temperature = 20
alpha = 0.5


train_loader, test_loader = get_data_loaders(batch_size=64)
_, optimizer_teacher, criterion_teacher = setup_teacher_model()


teacher = torch.load("Original_Models/main_model.pt", map_location=device)

# Set up the optimizer and loss function for teacher training
optimizer_teacher = optim.SGD(
    teacher.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
criterion_teacher = nn.CrossEntropyLoss()

student, optimizer_student, criterion_student = setup_student_model()

# Set up the optimizer and loss function for student training
optimizer_student = optim.SGD(
    student.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
criterion_student = nn.KLDivLoss()

train_student_with_distillation(teacher, student, train_loader, test_loader,
                                optimizer_student, criterion_teacher, criterion_student, temperature, alpha, 100)


torch.save(student, os.path.join("Original_Models", "distilled_vgg11.pt"))
