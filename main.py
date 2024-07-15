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
import requests
from pathlib import Path
from helper_functions import accuracy_fn
from timeit import default_timer as timer
from tqdm.auto import tqdm

if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = CowsDataset("/Users/safesonali/Desktop/DSI-2024/depth_processed",
                               "/Users/safesonali/Desktop/DSI-2024/bcs_dict.csv",
                               mode='gradangle',
                               transform=transform)

    train_size = math.floor(0.8*len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    BATCH_SIZE = 16

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=BATCH_SIZE,
                                  num_workers=1,
                                  shuffle=True)

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=BATCH_SIZE,
                                 num_workers=1,
                                 shuffle=False)

    train_features_batch, train_labels_batch = next(iter(train_dataloader))
    class_names = full_dataset.labels

    def print_train_time(start: float, end: float, device: torch.device = None):
        total_time = end - start
        print(f"Train time on {device}: {total_time:.3f} seconds")
        return total_time
    class CowBCSCNN(nn.Module):
        def __init__(self, input_channels):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Flatten(),
                nn.Linear(256 * 29 * 64, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 11)
            )

        def forward(self, xb):
            return self.network(xb)

    model_0 = CowBCSCNN(input_channels=1)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

    torch.manual_seed(23)
    train_time_start = timer()
    epochs = 1

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n-------")
        train_loss = 0
        for batch, (X, y) in enumerate(train_dataloader):
            model_0.train()
            y_pred = model_0(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 10 == 0:
                print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")
        train_loss /= len(train_dataloader)

        ### Validation
        test_loss, test_acc = 0, 0
        model_0.eval()
        with torch.inference_mode():
            for X, y in test_dataloader:
                test_pred = model_0(X)
                test_loss += loss_fn(test_pred, y)
                test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)
        print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

    train_time_end = timer()
    total_train_time_model_0 = print_train_time(start=train_time_start,
                                                end=train_time_end,
                                                device=str(next(model_0.parameters()).device))



