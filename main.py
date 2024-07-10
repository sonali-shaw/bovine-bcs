from dataset import *
import math
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    start_time = time.time()
    full_dataset = CowsDataset("/Users/safesonali/Desktop/DSI-2024/depth_processed",
                               "/Users/safesonali/Desktop/DSI-2024/bcs_dict.csv",
                               mode='gradangle',
                               transform=transform)
    end_time = time.time()
    print(f"time to create dataset: {end_time - start_time}")


    train_size = math.floor(0.8*len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=1,
                                  num_workers=1,
                                  shuffle=True)

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=1,
                                 num_workers=1,
                                 shuffle=False)

    class_names = full_dataset.labels
    BATCH_SIZE = 32


    def accuracy_fn(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()  # torch.eq() calculates where two tensors are equal
        acc = (correct / len(y_pred)) * 100
        return acc

    # Define model
    class CowBCSCNN(nn.Module):
        def __init__(self):
            super(CowBCSCNN, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.fc1 = nn.Linear(in_features=64 * 58 * 129, out_features=128)
            self.fc2 = nn.Linear(in_features=128, out_features=len(class_names))

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64 * 58 * 129)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x


    model = CowBCSCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print("beginning trainiing")
    start_training_time = time.time()

    train_loss, train_acc = 0, 0
    # model.to(device)
    for batch, (X, y) in enumerate(train_dataloader):
        model.train()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

    end_training_time = time.time()
    print(f"time to train: {end_training_time - start_training_time}")


    # Evaluation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Accuracy of the model on the test images: {100 * correct / total}%')

