import random
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Dataset
import os
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torchvision import transforms

@dataclass
class DataObject:
    video: np.ndarray
    label: int

class CowsDatasetOld(Dataset):
  def __init__(self,
               root_dir: Path | str,
               csv_file: Path | str,
               mode: str = 'depth',
               transform: bool = None
              ):

    self.root_dir = Path(root_dir) if not isinstance(root_dir, Path) else root_dir
    self.csv_file = Path(csv_file) if not isinstance(csv_file, Path) else csv_file

    self.dataset = []
    self.padded_imgs = []
    self.labels = ['200', '225', '250', '275', '300', '325', '350', '375', '400', '425', '450']
    self.labels = {label: idx for idx, label in enumerate(self.labels)} ## maps original labels to 0-9

    self.transform = transform

    if mode == 'permute':
        modes = [ 'depth', 'adjacent', 'contour', 'median', 'laplacian', 'gradangle']
    if type(mode) == tuple:
        modes = list(mode)

    def __pad_array_to_shape(a, target_shape):
        # each of the numpy arrays are different shapes so this pads them out to be the same shape
        pad_width = [(0, max(t - s, 0)) for s, t in zip(a.shape, target_shape)]
        padded_array = np.pad(a, pad_width, mode='constant', constant_values=0)
        slices = tuple(slice(0, t) for t in target_shape)
        padded_array = padded_array[slices]
        return padded_array

    def search_name(name):
        """ helper function for getting the name to search the csv file"""
        name_spl = name.split("_")
        return name_spl[0] + "_" + name_spl[1]

    def __get_lengths(arr):
        # returns the inner and outer largest dimesnions to be referenced for padding out the arrays
        largest_frame = 0
        largest_channel = 0
        for frame in arr:
            frame_length = len(frame)
            if frame_length > largest_frame:
                largest_frame = frame_length
            for channel in frame:
                channel_length = len(channel)
                if channel_length > largest_channel:
                    largest_channel = channel_length
        return largest_frame, largest_channel

    csv_df = pd.read_csv(self.csv_file)
    label_dict = {}

    for tup in csv_df.itertuples():
        filename, bcs = tup.filename, tup.BCSq
        filename = filename.split("_")
        filename = "_".join([filename[i] for i in (0, 1)])
        label_dict[filename] = bcs

    all_imgs = []
    corr_labels = [] # corresponding labels
    filenames = os.listdir(self.root_dir)
    for name in filenames:
        if name[0] != ".":
            file_np = np.load(os.path.join(self.root_dir, name), allow_pickle=True)
            label = label_dict[search_name(name)]
            mapped_lbl = self.labels[str(label)]
            if mode == 'permute':
                for i in range(50): # because there are 50 frames in each video
                    rand_mode = modes[random.randint(0, 5)]
                    all_imgs.append(file_np[rand_mode][i])
                    corr_labels.append(mapped_lbl)
            elif type(mode) == tuple:
                for i in range(50):
                    rand_mode = modes[[random.randint(0, len(modes)-1)]]
                    all_imgs.append(file_np[rand_mode][i])
                    corr_labels.append(mapped_lbl)
            else:
                video = np.array(file_np[mode])
                for img in video:
                    all_imgs.append(img)
                    corr_labels.append(mapped_lbl)

    frame_length, channel_length = __get_lengths(all_imgs)
    for i in range(len(all_imgs)):
        padded_img = __pad_array_to_shape(all_imgs[i], (frame_length, channel_length))
        self.dataset.append( (padded_img, corr_labels[i]) )

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    image, label = self.dataset[idx]
    if self.transform:
        item_transformed = self.transform(image)
        return item_transformed, label
    return image, label

if __name__ == '__main__':
    pass
    # start_time = time.time()
    full_dataset = CowsDatasetOld("/Users/safesonali/Desktop/DSI-2024/depth_processed",
                                   "/Users/safesonali/Desktop/DSI-2024/bcs_dict.csv",
                                   mode='depth',
                                transform=transforms.ToTensor())
    # print(full_dataset.state_dict())
    # end_time = time.time()
    # print(f"time taken: {end_time - start_time}")
    # # print(full_dataset[0][0])
    # print(len(full_dataset[0][0][0]))
    # print(len(full_dataset[0][0][0][0]))

    img, label = full_dataset[340]
    plt.imshow(img[0])
    plt.show()
    print(label)


    # for i in range(10):
    #     print("-------------------")
    #     print(full_dataset[i])

    # visualization
    # frames = []
    # for i in range(200):
    #     frames.append(full_dataset[i][0])
    #
    # fig, ax = plt.subplots()
    # def update_frame(frame_idx):
    #   frame = frames[frame_idx]
    #   ax.cla()
    #   img = ax.imshow(frame)
    #   return img
    #
    # animation = FuncAnimation(fig, update_frame, frames=len(frames), interval=50)
    # plt.show()

