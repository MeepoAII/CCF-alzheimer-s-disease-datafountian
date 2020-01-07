import torch

import torch.nn as nn
from torch import cat
import torch.nn.init as init
import math
import sys
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import time

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

import torch.nn.functional as F

from datetime import datetime
start = datetime.now()


# define my class to read data
class MyDataset(Dataset):
    def __init__(self, transform=None, img_path=None, label_path=None):
        self.transform = transform
        self.img_path = img_path
        self.label_path = label_path
        self.labels = pd.read_csv(self.label_path)
        self.images_file = h5py.File(self.img_path)
        self.images = self.images_file['data']

    def __getitem__(self, index):
        # 3d convolution 
        # so don't reshape
        return self.images[index], self.labels['label'][index]


    def __len__(self):
        return len(self.images)



class FirstNet(nn.Module):

    def __init__(self, f=8):
        super(FirstNet, self).__init__()

        self.conv = nn.Sequential()
        self.conv.add_module('conv1', nn.Conv3d(in_channels=1, out_channels=4 * f, kernel_size=1, stride=1, padding=0, dilation=1))
        self.conv.add_module('conv2', nn.InstanceNorm3d(num_features=4 * f))
        self.conv.add_module('conv3', nn.ReLU(inplace=True))
        self.conv.add_module('conv4', nn.MaxPool3d(kernel_size=3, stride=2))

        self.conv.add_module('conv5', nn.Conv3d(in_channels=4 * f, out_channels=32 * f, kernel_size=2, stride=1, padding=0, dilation=2))
        self.conv.add_module('conv6', nn.InstanceNorm3d(num_features=32 * f))
        self.conv.add_module('conv7', nn.ReLU(inplace=True))
        self.conv.add_module('conv8', nn.MaxPool3d(kernel_size=3, stride=2))

        self.conv.add_module('conv9', nn.Conv3d(in_channels=32 * f, out_channels=64 * f, kernel_size=2, stride=1, padding=2, dilation=2))
        self.conv.add_module('conv10', nn.InstanceNorm3d(num_features=64 * f))
        self.conv.add_module('conv11', nn.ReLU(inplace=True))
        self.conv.add_module('conv12', nn.MaxPool3d(kernel_size=3, stride=2))

        self.conv.add_module('conv13', nn.Conv3d(in_channels=64 * f, out_channels=64 * f, kernel_size=2, stride=1, padding=1, dilation=2))
        self.conv.add_module('conv14', nn.InstanceNorm3d(num_features=64 * f))
        self.conv.add_module('conv15', nn.ReLU(inplace=True))
        self.conv.add_module('conv16', nn.MaxPool3d(kernel_size=5, stride=2))

        self.fc = nn.Sequential()
        self.fc.add_module('fc1', nn.Linear(64 * f * 3 * 4 * 3, 1024))
        self.fc.add_module('dp1', nn.Dropout(0.3))
        self.fc.add_module('fc2', nn.Linear(1024, 3))


    def forward(self, x):
        z = self.conv(x)
        z = self.fc(z.view(x.shape[0], -1))
        return z



train_data = MyDataset(img_path='train.h5',
                 label_path='tran.csv')

train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=False)
print("Train data load success")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FirstNet(f=8)

print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)
model.to(device)

epochs = 200
steps = 0
running_loss = 0
print_every = 10
train_loss = []

for epoch in range(epochs):
    for inputs, labels in train_loader:
        print("Steps: ", steps)
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss.append(running_loss / len(train_loader))
    print(f"Epoch {epoch + 1} / {epochs}.."
          f"Train loss: {running_loss / print_every:.70f}..")

    running_loss = 0
    model.train()
torch.save(model, "firstnetmodel.pth")

plt.plot(train_loss, label='Training loss')
plt.legend(frameon=False)
plt.show()

stop = datetime.now()
print("Running time: ", stop-start)
