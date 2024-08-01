from archived.dataset import *
import math
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision import transforms
from timeit import default_timer as timer
from tqdm.auto import tqdm
import torchvision

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    start_load = timer()
    full_dataset = CowsDatasetOld("/Users/safesonali/Desktop/DSI-2024/depth_processed",
                               "/Users/safesonali/Desktop/DSI-2024/bcs_dict.csv",
                               mode='median',
                            transform=transform)
    end_load = timer()
    print(f"time to load data: {end_load-start_load}")

    train_size = math.floor(0.8*len(full_dataset))
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

    train_features_batch, train_labels_batch = next(iter(train_dataloader))

    flatten_model = nn.Flatten()
    x = train_features_batch[0]
    output = flatten_model(x)
    class_names = full_dataset.labels

    def print_train_time(start: float, end: float, device: torch.device = None):
        total_time = end - start
        print(f"Train time on {device}: {total_time:.3f} seconds")
        return total_time


    class CowBCSCNN(nn.Module):

        def __init__(self, input_channels):
            super(CowBCSCNN, self).__init__()
            self.resnet = torchvision.models.resnet18(pretrained=True)
            self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.resnet.fc = nn.Linear(512, 1)

        def forward(self, xb):
            return self.resnet(xb)

    model_0 = CowBCSCNN(input_channels=1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

    train_time_start = timer()
    torch.manual_seed(43)
    epochs = 15


    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch+1}\n-------")
        train_loss = 0

        for batch, (X, y) in enumerate(train_dataloader):
            start_batch = timer()
            model_0.train()
            y_pred = model_0(X)
            # y = y.view(-1, 1)
            loss = loss_fn(y_pred, y.float())
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end_batch = timer()

            if batch % 8 == 0:
                print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")

        train_loss /= len(train_dataloader)
        print(f"Train loss: {train_loss:.5f}")

        ### Validation
        test_loss = 0
        model_0.eval()

        with torch.inference_mode():

            for X, y in test_dataloader:
                test_pred = model_0(X)
                y = y.view(-1, 1)
                test_loss += loss_fn(test_pred, y.float())

            test_loss /= len(test_dataloader)
            print(f"Test loss: {test_loss:.5f}")

    train_time_end = timer()
    total_train_time_model_0 = print_train_time(start=train_time_start,
                                                end=train_time_end,
                                                device=str(next(model_0.parameters()).device))

    # testing
    def eval_model(model: torch.nn.Module,
                   data_loader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.Module,
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
        loss = 0
        model.eval()

        with torch.inference_mode():

            for X, y in data_loader:
                y_pred = model(X)
                y = y.view(-1, 1)
                loss += loss_fn(y_pred, y.float())

            loss /= len(data_loader)

        return {"model_name": model.__class__.__name__,  # only works when model was created with a class
                "model_loss": loss.item()}

    model_results = eval_model(model=model_0,
                                 data_loader=test_dataloader,
                                 loss_fn=loss_fn,
                                 device=device)
    print(model_results)

    # saving
    #
    # MODEL_PATH = Path("models")
    # MODEL_PATH.mkdir(parents=True,exist_ok=True)
    #
    # MODEL_NAME = "cowsCNNmodel-REGRESSION.pth"
    # MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    #
    # print(f"Saving model to: {MODEL_SAVE_PATH}")
    # torch.save(obj=model_0.state_dict(),f=MODEL_SAVE_PATH)




