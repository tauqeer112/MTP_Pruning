import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import vgg16
from utils import evaluate_accuracy
from utils import device
import gc
import numpy as np

# set random seed
torch.manual_seed(42)
np.random.seed(42)

def train_teacher(teacher, optimizer_teacher, criterion_teacher, train_loader, test_loader, n_epochs=5):
    # Train the teacher for n_epochs epochs
    teacher.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # Zero the parameter gradients
            optimizer_teacher.zero_grad()

            # Forward + backward + optimize
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = teacher(inputs)
            loss = criterion_teacher(outputs, labels)
            loss.backward()
            optimizer_teacher.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Teacher: Epoch {epoch+1}, Batch {i+1}: Loss {running_loss/100:.3f}')
                running_loss = 0.0

        # Evaluate teacher accuracy on test set
        accuracy = evaluate_accuracy(teacher,test_loader)
      
        print(f'Teacher: Epoch {epoch+1}, Test Accuracy: {accuracy:.2f}%')

    print('Finished Teacher Training')

    