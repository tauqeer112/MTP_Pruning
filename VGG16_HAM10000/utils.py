import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import vgg16 ,vgg11
import numpy as np
import torchvision
import torchvision.models as models
import copy
import os

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
# device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

# set random seed
torch.manual_seed(42)
np.random.seed(42)

import pandas as pd
from torch.utils.data import Dataset 
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

class HAMDataset(Dataset):
    def __init__(self, csv_file , root_dir, train=True, transforms = None):
        self.df = pd.read_csv(csv_file,index_col=0)
        if(train == True):
            self.df = self.df.iloc[:7000 , :]
        else:
            self.df = self.df.iloc[7000: , :]
        self.root_dir = root_dir
        self.transforms = transforms
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name_image = self.df.iloc[idx,0]
        image_filepath = os.path.join(self.root_dir,name_image)
        image = Image.open(image_filepath)
        if self.transforms:
            image = self.transforms(image)
        
        label = self.df.iloc[idx,-1]
        
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



def get_data_loaders(batch_size=64, num_workers=4):

    root_dir = 'ham1000-segmentation-and-classification/images'
    trainset = HAMDataset(csv_file='ham1000-segmentation-and-classification/file_label.csv',
                        root_dir = root_dir, train = True , transforms = transform)
    testset = HAMDataset(csv_file='ham1000-segmentation-and-classification/file_label.csv', 
                        root_dir=root_dir , train=False , transforms= transform)

    trainloader = DataLoader(trainset, batch_size=batch_size,pin_memory=False,
                                            shuffle=True, num_workers=16)
    testloader = DataLoader(testset, batch_size=batch_size,
                                            shuffle=False,pin_memory=True, num_workers=26)
    return trainloader , testloader

def setup_teacher_model():
    # Set up the VGG16 teacher model
    teacher = vgg16(pretrained=True)
    teacher.classifier[6] = nn.Linear(4096, 7)
    teacher = teacher.to(device)

    # Set up the optimizer and loss function for teacher training
    optimizer_teacher = optim.SGD(teacher.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4) 
    criterion_teacher = nn.CrossEntropyLoss()

    return teacher, optimizer_teacher, criterion_teacher



def setup_student_model():
    # Set up the smaller VGG16 student model
    student = vgg11(pretrained=True)
    student.classifier[6] = nn.Linear(4096, 7)
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
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            total_params += module.weight.nelement()
            nonzero_params += torch.sum(module.weight != 0)

    # Compute the sparsity of the model
    sparsity = 1 - (nonzero_params / total_params)
    print(f"Sparsity: {sparsity:.2%}")
    return sparsity