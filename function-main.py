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
from timeit import default_timer as timer
from tqdm.auto import tqdm

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    full_dataset = CowsDataset("/Users/safesonali/Desktop/DSI-2024/depth_processed",
                               "/Users/safesonali/Desktop/DSI-2024/bcs_dict.csv",
                               mode='gradangle',
                               transform=transform)

    train_size = math.floor(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    BATCH_SIZE = 32
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=BATCH_SIZE,
                                  num_workers=1,
                                  shuffle=True)

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=BATCH_SIZE,
                                 num_workers=1,
                                 shuffle=False)

    class_names = full_dataset.classes
    train_features_batch, train_labels_batch = next(iter(train_dataloader))

    torch.manual_seed(42)
    flatten_model = nn.Flatten()  # all nn modules function as a model (can do a forward pass)
    x = train_features_batch[0]
    output = flatten_model(x)


    def accuracy_fn(y_true, y_pred):
        """Calculates accuracy between truth labels and predictions.

        Args:
            y_true (torch.Tensor): Truth labels for predictions.
            y_pred (torch.Tensor): Predictions to be compared to predictions.

        Returns:
            [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
        """
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100
        return acc

    def print_train_time(start: float, end: float, device: torch.device = None):
        """Prints difference between start and end time.

        Args:
            start (float): Start time of computation (preferred in timeit format).
            end (float): End time of computation.
            device ([type], optional): Device that compute is running on. Defaults to None.

        Returns:
            float: time between start and end in seconds (higher is longer).
        """
        total_time = end - start
        print(f"Train time on {device}: {total_time:.3f} seconds")
        return total_time

    def train_step(model: torch.nn.Module,
                   data_loader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   accuracy_fn,
                   device: torch.device = device):
        train_loss, train_acc = 0, 0
        model.to(device)
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss
            train_acc += accuracy_fn(y_true=y,
                                     y_pred=y_pred.argmax(dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= len(data_loader)
        train_acc /= len(data_loader)
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

    def test_step(data_loader: torch.utils.data.DataLoader,
                  model: torch.nn.Module,
                  loss_fn: torch.nn.Module,
                  accuracy_fn,
                  device: torch.device = device):
        test_loss, test_acc = 0, 0
        model.to(device)
        model.eval()
        with torch.inference_mode():
            for X, y in data_loader:
                X, y = X.to(device), y.to(device)
                test_pred = model(X)
                test_loss += loss_fn(test_pred, y)
                test_acc += accuracy_fn(y_true=y,
                                        y_pred=test_pred.argmax(dim=1)  )
            test_loss /= len(data_loader)
            test_acc /= len(data_loader)
            print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")


    def eval_model(model: torch.nn.Module,
                   data_loader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.Module,
                   accuracy_fn,
                   device: torch.device = device):
        """Evaluates a given model on a given dataset.

        Args:
            model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
            data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
            loss_fn (torch.nn.Module): The loss function of model.
            accuracy_fn: An accuracy function to compare the models predictions to the truth labels.
            device (str, optional): Target device to compute on. Defaults to device.

        Returns:
            (dict): Results of model making predictions on data_loader.
        """
        loss, acc = 0, 0
        model.eval()
        with torch.inference_mode():
            for X, y in data_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss += loss_fn(y_pred, y)
                acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
            loss /= len(data_loader)
            acc /= len(data_loader)
        return {"model_name": model.__class__.__name__,
                "model_loss": loss.item(),
                "model_acc": acc}

    class CowModel(nn.Module):
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
        
    model = CowModel(input_channels=1)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(),lr=0.1)

    torch.manual_seed(42)

    train_time_start = timer()
    epochs = 3

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n---------")
        train_step(data_loader=train_dataloader,
                   model=model,
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   accuracy_fn=accuracy_fn)
        test_step(data_loader=test_dataloader,
                  model=model,
                  loss_fn=loss_fn,
                  accuracy_fn=accuracy_fn)

    train_time_end= timer()
    total_train_time = print_train_time(start=train_time_start,end=train_time_end,device=device)

    model_results = eval_model(model=model, data_loader=test_dataloader,
                                 loss_fn=loss_fn, accuracy_fn=accuracy_fn)
    print(f"model results: {model_results}")

    # save model

    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True )
    MODEL_NAME = "03_pytorch_computer_vision_model_2.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(),
               f=MODEL_SAVE_PATH)
