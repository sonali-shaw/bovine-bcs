from dataset import *
import math
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from PIL import Image
from pathlib import Path
from helper_functions import accuracy_fn
from timeit import default_timer as timer
from tqdm.auto import tqdm
import torchvision
from torchmetrics import Accuracy

# THIS IS CLASSIFICATION

class CowsDataModule(LightningDataModule):
    def __init__(self, data_dir, bcs_csv, batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.bcs_csv = bcs_csv
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])


    def setup(self, stage=None):
        full_dataset = CowsDatasetOld(self.data_dir, self.bcs_csv, mode='gradangle', transform=self.transform)
        train_size = math.floor(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        self.train_data, self.test_data = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=1, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=1, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=1, shuffle=False)

class CowBCSCNN(LightningModule):
    def __init__(self, input_channels=1, learning_rate=0.01):
        super(CowBCSCNN, self).__init__()
        self.save_hyperparameters()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, 11)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task='multiclass', num_classes=11)
        self.val_acc = Accuracy(task='multiclass', num_classes=11)
        self.test_acc = Accuracy(task='multiclass', num_classes=11)

    def forward(self, xb):
        return self.resnet(xb)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y = y.squeeze().long()
        y_pred = self(X)
        print(f"Training Step - y_pred shape: {y_pred.shape}, y shape: {y.shape}")
        loss = self.loss_fn(y_pred, y)
        acc = self.train_acc(y_pred.argmax(dim=1), y)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y = y.squeeze().long()
        y_pred = self(X)
        print(f"Validation Step - y_pred shape: {y_pred.shape}, y shape: {y.shape}")
        loss = self.loss_fn(y_pred, y)
        acc = self.val_acc(y_pred.argmax(dim=1), y)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        X, y = batch
        y = y.squeeze().long()
        y_pred = self(X)
        print(f"Test Step - y_pred shape: {y_pred.shape}, y shape: {y.shape}")
        loss = self.loss_fn(y_pred, y)
        acc = self.test_acc(y_pred.argmax(dim=1), y)
        self.log('test_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('test_acc', acc, prog_bar=True, on_step=True, on_epoch=True)

    def on_train_epoch_end(self):
        self.log('train_acc_epoch', self.train_acc.compute())
        self.train_acc.reset()

    def on_validation_epoch_end(self):
        self.log('val_acc_epoch', self.val_acc.compute())
        self.val_acc.reset()

    def on_test_epoch_end(self):
        self.log('test_acc_epoch', self.test_acc.compute())
        self.test_acc.reset()

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.hparams.learning_rate)

if __name__ == '__main__':
    try:
        data_module = CowsDataModule(data_dir="/Users/safesonali/Desktop/DSI-2024/depth_processed",
                                     bcs_csv="/Users/safesonali/Desktop/DSI-2024/bcs_dict.csv")

        model = CowBCSCNN()

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='models/',
            filename='cowbcs-cnn-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            mode='min',
        )

        logger = TensorBoardLogger("tb_logs", name="cowbcs_cnn")

        trainer = Trainer(max_epochs=5, callbacks=[checkpoint_callback], logger=logger)

        start_train = timer()
        trainer.fit(model, data_module)
        end_train = timer()

        total_train_time = end_train - start_train
        print(f"Total training time: {total_train_time:.3f} seconds")

        results = trainer.test(model, dataloaders=data_module.test_dataloader())
        print(results)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
