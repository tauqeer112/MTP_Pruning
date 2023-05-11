import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import vgg16
from utils import device
import numpy as np

# set random seed
torch.manual_seed(42)
np.random.seed(42)


def train_student_with_distillation(teacher, student, train_loader, test_loader, optimizer_student, criterion_teacher, criterion_student, temperature=4, alpha=0.7, num_epochs=20):
    # Set teacher model to evaluation mode
    teacher.eval()
    student.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # Zero the parameter gradients
            optimizer_student.zero_grad()

            # Forward pass through teacher and student models
            inputs = inputs.to(device)
            labels = labels.to(device)
            teacher_outputs = teacher(inputs).detach()
            student_outputs = student(inputs)

            # Compute the knowledge distillation loss
            loss = alpha * criterion_student(torch.log_softmax(student_outputs/temperature, dim=1),
                                             torch.softmax(teacher_outputs/temperature, dim=1)) \
                + (1 - alpha) * criterion_teacher(student_outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer_student.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:
                print(
                    f'Student: Epoch {epoch+1}, Batch {i+1}: Loss {running_loss/100:.3f}')
                running_loss = 0.0

        # Evaluate student accuracy on test set
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = student(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(
            f'Student: Epoch {epoch+1}, Test Accuracy: {correct/total*100:.2f}%')

    # Evaluate teacher accuracy on test set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = teacher(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Teacher: Test Accuracy: {correct/total*100:.2f}%')

    # Evaluate student accuracy on test set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = student(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Student: Test Accuracy: {correct/total*100:.2f}%')
