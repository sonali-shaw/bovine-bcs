from dataset import *
from torch.utils.data import DataLoader

data = CowsDataset("/Volumes/Samsung USB/depth_processed", "/Volumes/Samsung USB/bcs_dict.csv")
print(len(data))
print(data.__getitem__(2))

# # figure out train/test split
#
# train_data = None
# test_data = None
#
# train_dataloader = DataLoader(dataset=train_data,
#                               batch_size=1,
#                               num_workers=1,
#                               shuffle=True)
#
# test_dataloader = DataLoader(dataset=test_data,
#                              batch_size=1,
#                              num_workers=1,
#                              shuffle=False)