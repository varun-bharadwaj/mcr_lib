import torchvision
import torch.optim as optim

from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from lib import Network, LossFunc

train_data = torchvision.datasets.CIFAR10('/cifar-10-batches-py',train=True, download=True, transform=transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10('/cifar-10-batches-py',train=False, download=True, transform=transforms.ToTensor())

batch_size = 100
trainloader = DataLoader(train_data, batch_size=batch_size, drop_last=True, num_workers=4)

model = Network()

optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=2)
loss_fn = LossFunc()
num_epochs = 20

for epoch in tqdm(range(num_epochs)):
    # Forward pass
    for (batch_imgs, batch_lbls) in trainloader:
        batch_imgs = batch_imgs.reshape((batch_size, 3072))
        model = model.float()
        outputs =  model(batch_imgs)
        loss = loss_fn(outputs, batch_lbls)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(range(len(loss_fn.rs)), loss_fn.rs)
plt.xlabel('num_iterations')
plt.ylabel('Coding Rate')
ax.set_xscale("log", base=10);
plt.show()