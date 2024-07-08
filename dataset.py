import random
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Dataset
import os
import csv
from dataclasses import dataclass

# folder_path = "/Volumes/Samsung USB/depth_processed"

@dataclass
class DataObject:
    video: np.ndarray
    label: int

class CowsDataset(Dataset):
  def __init__(self, root_dir, csv_file, mode='depth', permute=False):
    self.padded_imgs = []
    modes = [ 'depth', 'adjacent', 'contour', 'median', 'laplacian', 'gradangle']

    def __pad_array_to_shape(a, target_shape):
      pad_width = [(0, max(t - s, 0)) for s, t in zip(a.shape, target_shape)]
      padded_array = np.pad(a, pad_width, mode='constant', constant_values=0)
      slices = tuple(slice(0, t) for t in target_shape)
      padded_array = padded_array[slices]
      return padded_array
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

    # get numpy arrays from root directory and add to dataset
    largest_channel = 0
    largest_frame = 0
    filenames = os.listdir(root_dir)
    for name in filenames:
      if name[0] != ".":
        file_np = np.load(os.path.join(root_dir, name), allow_pickle=True)
        label = label_dict[search_name(name)]

        if permute:
          video = []
          for i in range(50): # because there are 50 frames in each video
            rand_mode = modes[random.randint(0, 5)]
            video.append(file_np[rand_mode][i])
        else:
          video = np.array(file_np[mode])

        for frame in video:
          if len(frame) > largest_frame:
            largest_frame = len(frame)
          if len(frame[0]) > largest_channel:
            largest_channel = len(frame[0])
          tup = (__pad_array_to_shape(frame, (largest_frame, largest_channel)), label)
          self.padded_imgs.append(tup)

  def __len__(self):
    return len(self.padded_imgs)

  def __getitem__(self, idx):
    item = self.padded_imgs[idx]
    return torch.from_numpy(item[0]), item[1]

