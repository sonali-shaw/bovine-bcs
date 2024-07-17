import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningDataModule
from dataset import *
import math
from helper_functions import accuracy_fn


class CowsDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, transform):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):
        full_dataset = CowsDataset("/Users/safesonali/Desktop/DSI-2024/depth_processed",
                               "/Users/safesonali/Desktop/DSI-2024/bcs_dict.csv",
                               mode='gradangle',
                               transform=transforms)
        train_size = math.floor(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        self.train_data, self.test_data = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=1, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=1, shuffle=False)

class CowBCSCNN(pl.LightningModule):
    def __init__(self, input_channels, lr):
        super().__init__()
        self.save_hyperparameters()
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
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, xb):
        return self.network(xb)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)
        acc = accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)
        acc = accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
        return optimizer

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    data_module = CowsDataModule(data_dir="/Users/safesonali/Desktop/DSI-2024/depth_processed",
                                 batch_size=32,
                                 transform=transform)

    model = CowBCSCNN(input_channels=1, lr=0.1)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='models/',
        filename='cowsCNNmodel-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    trainer = Trainer(max_epochs=3, callbacks=[checkpoint_callback], accelerator="auto")

    trainer.fit(model, data_module)

    trainer.save_checkpoint("models/cowsCNNmodel-final.ckpt")

if __name__ == '__main__':
    main()
