import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

def __pad_array_to_shape(a, target_shape):
    pad_width = [(0, max(t - s, 0)) for s, t in zip(a.shape, target_shape)]
    padded_array = np.pad(a, pad_width, mode='constant', constant_values=0)
    slices = tuple(slice(0, t) for t in target_shape)
    padded_array = padded_array[slices]
    return padded_array

loaded_np = np.load("/Volumes/Samsung USB/depth_processed/8008_20210122_adj10_floor5000.npz", allow_pickle=True)
modes = ['depth', 'adjacent', 'contour', 'median', 'laplacian', 'gradangle']

frames = []

for i in range(50):
    rand_mode = modes[random.randint(0, 5)]
    frames.append(np.array(loaded_np[rand_mode][i]))

# frames = np.array(frames_)
# print(frames[0])
#
# frames = np.array(loaded_np['depth'])
# print(len(frames))
# print(len(frames[0]))
#
# plt.imshow(frames_[0])
# plt.show()


# visualization
fig, ax = plt.subplots()
def update_frame(frame_idx):
  frame = frames[frame_idx]
  ax.cla()
  img = ax.imshow(frame)
  return img

animation = FuncAnimation(fig, update_frame, frames=len(frames), interval=100)
plt.show()

