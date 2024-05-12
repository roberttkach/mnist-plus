import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data.load import get_data
from dataset import MNISTPlusDataset, transform
from model import Net
import random
import os

get_data()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainset = MNISTPlusDataset(csv_file='./data/mnist+.csv', transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

testset = MNISTPlusDataset(csv_file='./data/mnist+.csv', transform=transform)
testloader = DataLoader(testset, batch_size=128, shuffle=True)

net = Net()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

for epoch in range(10):
    net.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    scheduler.step()

    net.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    accuracy_train = correct_train / total_train
    accuracy_test = correct_test / total_test

    if not os.path.exists('./saves'):
        os.makedirs('./saves')

    torch.save(net.state_dict(), f'./saves/weights/classification-{round(accuracy_test * 100, 2)}.pth')
    print(
        f'Epoch {epoch + 1} completed, loss: {round(running_loss / len(trainloader), 5)}, '
        f'accuracy-train: {round(accuracy_train, 3)}%, accuracy-test: {round(accuracy_test, 3)}%')

    dataiter = iter(testloader)
    images, labels = next(iter(dataiter))

    indices = random.sample(range(images.size(0)), 9)

    images = images[indices]
    labels = labels[indices]

    outputs = net(images.to(device))
    _, predicted = torch.max(outputs.data, 1)

    images = images.cpu()
    predicted = predicted.cpu()

    fig, axs = plt.subplots(3, 3)
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title(f"True: {labels[i]}, Pred: {predicted[i]}")
        ax.axis('off')

    plt.tight_layout()

    os.makedirs('./saves/plots', exist_ok=True)
    plt.savefig(f'./saves/plots/{epoch + 1}.png')
