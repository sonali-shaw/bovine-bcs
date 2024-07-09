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

# folder_path = "/Volumes/Samsung USB/depth_processed"

@dataclass
class DataObject:
    video: np.ndarray
    label: int

class CowsDataset(Dataset):
  def __init__(self,
               root_dir: Path | str,
               csv_file: Path | str,
               mode: str = 'depth',
               permute: bool = False,
               permuted_types: tuple = ()
              ):

    self.root_dir = Path(root_dir) if not isinstance(root_dir, Path) else root_dir
    self.csv_file = Path(csv_file) if not isinstance(csv_file, Path) else csv_file

    self.dataset = []
    self.padded_imgs = []
    if mode == 'permute':
        modes = [ 'depth', 'adjacent', 'contour', 'median', 'laplacian', 'gradangle']
    if type(mode) == tuple:
        modes = list(mode)

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
    def __get_frame_length(arr):
        biggest = 0
        for i in range(len(arr)):
            var = len(arr[i])
            if var > biggest:
                biggest = var
        return biggest
    def __get_channel_length(arr):
        biggest = 0
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                var = len(arr[i][j])
                if var > biggest:
                    biggest = var
        return biggest

    csv_df = pd.read_csv(self.csv_file)
    label_dict = {}

    for tup in csv_df.itertuples():
        filename, bcs = tup.filename, tup.BCSq
        filename = filename.split("_")
        filename = "_".join([filename[i] for i in (0, 1)])
        label_dict[filename] = bcs

    all_imgs = []
    corr_labels = []
    filenames = os.listdir(self.root_dir)
    for name in filenames:
        if name[0] != ".":
            file_np = np.load(os.path.join(self.root_dir, name), allow_pickle=True)
            label = label_dict[search_name(name)]
            if mode == 'permute':
                for i in range(50): # because there are 50 frames in each video
                    rand_mode = modes[random.randint(0, 5)]
                    all_imgs.append(file_np[rand_mode][i])
                    corr_labels.append(label)
            else:
                video = np.array(file_np[mode])
                for img in video:
                    all_imgs.append(img)
                    corr_labels.append(label)

    channel_length = __get_channel_length(all_imgs)
    frame_length = __get_frame_length(all_imgs)
    for i in range(len(all_imgs)):
        padded_img = __pad_array_to_shape(all_imgs[i], (frame_length, channel_length))
        self.dataset.append( (padded_img, corr_labels[i]) )

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    return torch.from_numpy(item[0]), item[1]

root_dir_path = Path("/Volumes/Samsung USB/depth_processed")
csv_path = Path("/Volumes/Samsung USB/bcs_dict.csv")

start_time = time.time()
full_dataset = CowsDataset(root_dir_path, csv_path, mode='permute')
end_time = time.time()
print(f"time taken: {end_time - start_time}")

frames = []
for i in range(200):
    frames.append(full_dataset[i][0])

fig, ax = plt.subplots()
def update_frame(frame_idx):
  frame = frames[frame_idx]
  ax.cla()
  img = ax.imshow(frame)
  return img

animation = FuncAnimation(fig, update_frame, frames=len(frames), interval=50)
plt.show()

