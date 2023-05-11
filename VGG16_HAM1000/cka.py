from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
from torch_cka import CKA
import torch
from utils import get_data_loaders
import os
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from utils import device, HAMDataset, transform
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# set random seed
torch.manual_seed(42)
np.random.seed(42)


batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train_loader , test_loader = get_data_loaders(batch_size=8)

root_dir = 'ham1000-segmentation-and-classification/images'
trainset = HAMDataset(csv_file='ham1000-segmentation-and-classification/file_label.csv',
                      root_dir=root_dir, train=True, transforms=transform)
testset = HAMDataset(csv_file='ham1000-segmentation-and-classification/file_label.csv',
                     root_dir=root_dir, train=False, transforms=transform)

train_loader = DataLoader(trainset, batch_size=batch_size, pin_memory=False,
                          shuffle=True, num_workers=16)
test_loader = DataLoader(testset, batch_size=batch_size,
                         shuffle=False, pin_memory=False, num_workers=26)


# Use the SubsetRandomSampler to create a subset of the test loader
# subset_sampler = SubsetRandomSampler(range(1000))
# subset_loader = DataLoader(test_loader.dataset, batch_size=16, sampler=subset_sampler)


path1 = ""  # path to first model
path2 = ""  # path to second model
title = ""  # title on image
filename = ""  # name of png file

model1 = torch.load(path1, map_location=device)
model2 = torch.load(path2, map_location=device)

base_dir_fig = "CKA_plots"

cka = CKA(model1, model1,
          model1_layers=layers,
          model2_layers=layers,
          device=device)
cka.compare(test_loader)
result = cka.export()
matrix = result["CKA"].numpy()

plt.figure(figsize=(15, 10))
ax = sns.heatmap(matrix)
save_path = os.path.join(base_dir_fig, filename)
plt.title(title)
plt.savefig(save_path)
