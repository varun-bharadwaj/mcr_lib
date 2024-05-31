import os

import torchvision
import torch.optim as optim

from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

from src.mcr_lib import loss
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(3072, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        
    def forward(self, x):
        logits = self.layers(x)
        return logits

train_data = torchvision.datasets.CIFAR10('/cifar-10-batches-py',train=True, download=True, transform=transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10('/cifar-10-batches-py',train=False, download=True, transform=transforms.ToTensor())

batch_size = 100
trainloader = DataLoader(train_data, batch_size=batch_size, drop_last=True)

model = Network()

optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=2)

num_epochs = 20

for epoch in tqdm(range(num_epochs)):
    # Forward pass
    
    for (batch_imgs, batch_lbls) in trainloader:
        batch_imgs = batch_imgs.reshape((batch_size, 3072))
        model = model.float()
        outputs =  model(batch_imgs)
        loss_val = loss(outputs, batch_lbls)
        # Backward and optimize
        optimizer.zero_grad()
        optimizer.step()
