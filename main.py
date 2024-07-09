from dataset import *
import math
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

start_time = time.time()

csv_path = Path("/Volumes/Samsung USB/bcs_dict.csv")

full_dataset = CowsDataset("/Volumes/Samsung USB/depth_processed", csv_path, mode='gradangle', permute=False)

end_time = time.time()
print(f"elapsed time: {end_time - start_time}")

frames = []
for i in range(50):
  frames.append(full_dataset[i][0])


fig, ax = plt.subplots()
def update_frame(frame_idx):
  frame = frames[frame_idx]
  ax.cla()
  img = ax.imshow(frame)
  return img

animation = FuncAnimation(fig, update_frame, frames=len(frames), interval=50)
plt.show()

#
#
# train_size = math.floor(0.8*len(full_dataset))
# test_size = len(full_dataset) - train_size
#
# train_data, test_data = torch.utils.data.random_split(full_dataset, [train_size, test_size])
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


