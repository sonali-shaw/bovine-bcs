# import torch
# from torch import nn
# # import torchvision
# # from torchvision import datasets
# # from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

data = np.load('/Volumes/Samsung USB/depth_processed/120_20210119_adj10_floor5000.npz', allow_pickle=True)

  # random_frame_ids
  # depth
  # adjacent
  # contour
  # median
  # laplacian
  # gradangle

# print(data['depth'])
image1 = data['depth']
frames = data['random_frame_ids']
image = image1[46]
print(data['adjacent'][0].shape)
plt.imshow(data['adjacent'][0])
plt.show()