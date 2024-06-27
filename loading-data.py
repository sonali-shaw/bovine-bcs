import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os
import pandas as pd
import csv

# folder_path = "/Volumes/Samsung USB/depth_processed"

class DataObject():
  def __init__(self, video, label):
    self.video = video
    self.label = label

class CowsDataset(Dataset):
  def __init__(self, root_dir, csv_file, mode='depth'):
    self.data = []

    def search_name(name):
      """ helper function for getting the name to search the csv file"""
      name_spl = name.split("_")
      return name_spl[0] + "_" + name_spl[1]

    # make dict of labels
    label_dict = {}
    with open(csv_file, 'r') as csvfile:
      reader = csv.reader(csvfile)
      next(reader, None)
      for row in reader:
        key, value = row[0], row[1]
        key_spl = key.split("_")
        key_to_add = key_spl[0] + "_" + key_spl[1]
        if key_to_add in label_dict:
          raise Exception("Repeated keys for labeling, check file names and column one of the csv. "
                          "LABELING WILL BE WRONG")
        label_dict[key_to_add] = value

    # get numpy arrays from root directory and add to data_dict
    filenames = os.listdir(root_dir)
    for name in filenames:
      if name[0] != ".":
        video_np = np.load(os.path.join(root_dir, name), allow_pickle=True)
        video = video_np[mode]

        try:
          label = label_dict[search_name(name)]
        except:
          print(f"No label found in the csv for {name}")

        new_item = DataObject(video, label)
        self.data.append(new_item)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    item = self.data[idx]
    return item.video, item.label

data = CowsDataset("/Volumes/Samsung USB/depth_processed", "/Volumes/Samsung USB/bcs_dict.csv")
# print(len(data))
print(data.__getitem__(2))
