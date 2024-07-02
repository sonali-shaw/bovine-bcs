from dataset import *
import math
from torch.utils.data import DataLoader

full_dataset = CowsDataset("/Volumes/Samsung USB/depth_processed", "/Volumes/Samsung USB/bcs_dict.csv")

# 70:30 split
train_size = math.floor(0.7*len(full_dataset))
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

