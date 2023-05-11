from trains import train_teacher
from utils import get_data_loaders,setup_teacher_model, setup_student_model
import torch.nn as nn
import torch.optim as optim
import torch 
import os
from Knowledge_Distillation import train_student_with_distillation
import numpy as np

# set random seed
torch.manual_seed(42)
np.random.seed(42)

temperature = 15
alpha = 0.7


train_loader, test_loader = get_data_loaders(batch_size=64)
_, optimizer_teacher, criterion_teacher = setup_teacher_model()

teacher = torch.load("Original_Models/main_model.pt")

student, optimizer_student, criterion_student = setup_student_model()



train_student_with_distillation(teacher , student , train_loader , test_loader , optimizer_student , criterion_teacher , criterion_student, temperature, alpha , 15)


torch.save(student , os.path.join("Original_Models", "distilled_vgg11.pt"))


