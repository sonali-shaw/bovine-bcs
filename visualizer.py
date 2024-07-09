import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd

loaded_np = np.load("/Volumes/Samsung USB/depth_processed/120_20210119_adj10_floor5000.npz", allow_pickle=True)
frames = np.array(loaded_np['depth'])
# print(loaded_np[0])


# fig, ax = plt.subplots()
# def update_frame(frame_idx):
#   frame = frames[frame_idx]
#   ax.cla()
#   img = ax.imshow(frame)
#   return img
#
# animation = FuncAnimation(fig, update_frame, frames=len(frames), interval=100)
# plt.show()