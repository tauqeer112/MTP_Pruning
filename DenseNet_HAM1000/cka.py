from torch.utils.data import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler
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
from utils import device
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# set random seed
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_loader, test_loader = get_data_loaders(batch_size=16)


# Use the SubsetRandomSampler to create a subset of the test loader
# subset_sampler = SubsetRandomSampler(range(500))
# subset_loader = DataLoader(
#     test_loader.dataset, batch_size=8, sampler=subset_sampler)

layers = ['features.conv0', 'features.denseblock1.denselayer1.conv1', 'features.denseblock1.denselayer1.conv2', 'features.denseblock1.denselayer2.conv1', 'features.denseblock1.denselayer2.conv2', 'features.denseblock1.denselayer3.conv1', 'features.denseblock1.denselayer3.conv2', 'features.denseblock1.denselayer4.conv1', 'features.denseblock1.denselayer4.conv2', 'features.denseblock1.denselayer5.conv1', 'features.denseblock1.denselayer5.conv2', 'features.denseblock1.denselayer6.conv1', 'features.denseblock1.denselayer6.conv2', 'features.transition1.conv', 'features.denseblock2.denselayer1.conv1', 'features.denseblock2.denselayer1.conv2', 'features.denseblock2.denselayer2.conv1', 'features.denseblock2.denselayer2.conv2', 'features.denseblock2.denselayer3.conv1', 'features.denseblock2.denselayer3.conv2', 'features.denseblock2.denselayer4.conv1', 'features.denseblock2.denselayer4.conv2', 'features.denseblock2.denselayer5.conv1', 'features.denseblock2.denselayer5.conv2', 'features.denseblock2.denselayer6.conv1', 'features.denseblock2.denselayer6.conv2', 'features.denseblock2.denselayer7.conv1', 'features.denseblock2.denselayer7.conv2', 'features.denseblock2.denselayer8.conv1', 'features.denseblock2.denselayer8.conv2', 'features.denseblock2.denselayer9.conv1', 'features.denseblock2.denselayer9.conv2', 'features.denseblock2.denselayer10.conv1', 'features.denseblock2.denselayer10.conv2', 'features.denseblock2.denselayer11.conv1', 'features.denseblock2.denselayer11.conv2', 'features.denseblock2.denselayer12.conv1', 'features.denseblock2.denselayer12.conv2', 'features.transition2.conv', 'features.denseblock3.denselayer1.conv1', 'features.denseblock3.denselayer1.conv2', 'features.denseblock3.denselayer2.conv1', 'features.denseblock3.denselayer2.conv2', 'features.denseblock3.denselayer3.conv1', 'features.denseblock3.denselayer3.conv2', 'features.denseblock3.denselayer4.conv1', 'features.denseblock3.denselayer4.conv2', 'features.denseblock3.denselayer5.conv1', 'features.denseblock3.denselayer5.conv2', 'features.denseblock3.denselayer6.conv1', 'features.denseblock3.denselayer6.conv2', 'features.denseblock3.denselayer7.conv1', 'features.denseblock3.denselayer7.conv2', 'features.denseblock3.denselayer8.conv1', 'features.denseblock3.denselayer8.conv2', 'features.denseblock3.denselayer9.conv1', 'features.denseblock3.denselayer9.conv2', 'features.denseblock3.denselayer10.conv1', 'features.denseblock3.denselayer10.conv2', 'features.denseblock3.denselayer11.conv1', 'features.denseblock3.denselayer11.conv2', 'features.denseblock3.denselayer12.conv1', 'features.denseblock3.denselayer12.conv2', 'features.denseblock3.denselayer13.conv1', 'features.denseblock3.denselayer13.conv2', 'features.denseblock3.denselayer14.conv1', 'features.denseblock3.denselayer14.conv2', 'features.denseblock3.denselayer15.conv1', 'features.denseblock3.denselayer15.conv2', 'features.denseblock3.denselayer16.conv1', 'features.denseblock3.denselayer16.conv2', 'features.denseblock3.denselayer17.conv1', 'features.denseblock3.denselayer17.conv2', 'features.denseblock3.denselayer18.conv1', 'features.denseblock3.denselayer18.conv2', 'features.denseblock3.denselayer19.conv1', 'features.denseblock3.denselayer19.conv2', 'features.denseblock3.denselayer20.conv1', 'features.denseblock3.denselayer20.conv2', 'features.denseblock3.denselayer21.conv1', 'features.denseblock3.denselayer21.conv2', 'features.denseblock3.denselayer22.conv1', 'features.denseblock3.denselayer22.conv2', 'features.denseblock3.denselayer23.conv1', 'features.denseblock3.denselayer23.conv2',
          'features.denseblock3.denselayer24.conv1', 'features.denseblock3.denselayer24.conv2', 'features.denseblock3.denselayer25.conv1', 'features.denseblock3.denselayer25.conv2', 'features.denseblock3.denselayer26.conv1', 'features.denseblock3.denselayer26.conv2', 'features.denseblock3.denselayer27.conv1', 'features.denseblock3.denselayer27.conv2', 'features.denseblock3.denselayer28.conv1', 'features.denseblock3.denselayer28.conv2', 'features.denseblock3.denselayer29.conv1', 'features.denseblock3.denselayer29.conv2', 'features.denseblock3.denselayer30.conv1', 'features.denseblock3.denselayer30.conv2', 'features.denseblock3.denselayer31.conv1', 'features.denseblock3.denselayer31.conv2', 'features.denseblock3.denselayer32.conv1', 'features.denseblock3.denselayer32.conv2', 'features.transition3.conv', 'features.denseblock4.denselayer1.conv1', 'features.denseblock4.denselayer1.conv2', 'features.denseblock4.denselayer2.conv1', 'features.denseblock4.denselayer2.conv2', 'features.denseblock4.denselayer3.conv1', 'features.denseblock4.denselayer3.conv2', 'features.denseblock4.denselayer4.conv1', 'features.denseblock4.denselayer4.conv2', 'features.denseblock4.denselayer5.conv1', 'features.denseblock4.denselayer5.conv2', 'features.denseblock4.denselayer6.conv1', 'features.denseblock4.denselayer6.conv2', 'features.denseblock4.denselayer7.conv1', 'features.denseblock4.denselayer7.conv2', 'features.denseblock4.denselayer8.conv1', 'features.denseblock4.denselayer8.conv2', 'features.denseblock4.denselayer9.conv1', 'features.denseblock4.denselayer9.conv2', 'features.denseblock4.denselayer10.conv1', 'features.denseblock4.denselayer10.conv2', 'features.denseblock4.denselayer11.conv1', 'features.denseblock4.denselayer11.conv2', 'features.denseblock4.denselayer12.conv1', 'features.denseblock4.denselayer12.conv2', 'features.denseblock4.denselayer13.conv1', 'features.denseblock4.denselayer13.conv2', 'features.denseblock4.denselayer14.conv1', 'features.denseblock4.denselayer14.conv2', 'features.denseblock4.denselayer15.conv1', 'features.denseblock4.denselayer15.conv2', 'features.denseblock4.denselayer16.conv1', 'features.denseblock4.denselayer16.conv2', 'features.denseblock4.denselayer17.conv1', 'features.denseblock4.denselayer17.conv2', 'features.denseblock4.denselayer18.conv1', 'features.denseblock4.denselayer18.conv2', 'features.denseblock4.denselayer19.conv1', 'features.denseblock4.denselayer19.conv2', 'features.denseblock4.denselayer20.conv1', 'features.denseblock4.denselayer20.conv2', 'features.denseblock4.denselayer21.conv1', 'features.denseblock4.denselayer21.conv2', 'features.denseblock4.denselayer22.conv1', 'features.denseblock4.denselayer22.conv2', 'features.denseblock4.denselayer23.conv1', 'features.denseblock4.denselayer23.conv2', 'features.denseblock4.denselayer24.conv1', 'features.denseblock4.denselayer24.conv2', 'features.denseblock4.denselayer25.conv1', 'features.denseblock4.denselayer25.conv2', 'features.denseblock4.denselayer26.conv1', 'features.denseblock4.denselayer26.conv2', 'features.denseblock4.denselayer27.conv1', 'features.denseblock4.denselayer27.conv2', 'features.denseblock4.denselayer28.conv1', 'features.denseblock4.denselayer28.conv2', 'features.denseblock4.denselayer29.conv1', 'features.denseblock4.denselayer29.conv2', 'features.denseblock4.denselayer30.conv1', 'features.denseblock4.denselayer30.conv2', 'features.denseblock4.denselayer31.conv1', 'features.denseblock4.denselayer31.conv2', 'features.denseblock4.denselayer32.conv1', 'features.denseblock4.denselayer32.conv2', 'classifier']


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
