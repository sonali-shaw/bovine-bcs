import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def pad_array_to_shape(a, target_shape):
    pad_width = [(0, max(t - s, 0)) for s, t in zip(a.shape, target_shape)]
    padded_array = np.pad(a, pad_width, mode='constant', constant_values=0)
    slices = tuple(slice(0, t) for t in target_shape)
    padded_array = padded_array[slices]
    return padded_array

def get_frame_length(arr):
    biggest = 0
    for i in range(len(arr)):
        var = len(arr[i])
        if var > biggest:
            biggest = var
    return biggest

def get_channel_length(arr):
    biggest = 0
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            var = len(arr[i][j])
            if var > biggest:
                biggest = var
    return biggest

loaded_np = np.load("/Volumes/Samsung USB/depth_processed/8008_20210122_adj10_floor5000.npz", allow_pickle=True)
frames = np.array(loaded_np['depth'])

new_frames = []
channel_length = get_channel_length(frames)
frame_length = get_frame_length(frames)

print(f"frame length: {frame_length}")
print(f"channel length: {channel_length}")

for frame in frames:
    new_frames.append(pad_array_to_shape(frame, (frame_length, channel_length)))

print(type(np.array(new_frames)))

#
fig, ax = plt.subplots()
def update_frame(frame_idx):
  frame = new_frames[frame_idx]
  ax.cla()
  img = ax.imshow(frame)
  return img

animation = FuncAnimation(fig, update_frame, frames=len(new_frames), interval=50)
plt.show()
