import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
import os
import torch


  # random_frame_ids
  # depth
  # adjacent
  # contour
  # median
  # laplacian
  # gradangle

# # print(data['depth'])
# image1 = data['depth']
# frames = data['random_frame_ids']
# image = image1[46]
# print(data['adjacent'][0].shape)
# plt.imshow(data['adjacent'][0])
# plt.show()




# def search_name(name):
#     """ helper function for getting the name to search the csv file"""
#     name_spl = name.split("_")
#     return name_spl[0] + "_" + name_spl[1]
#
# root_dir = "/Volumes/Samsung USB/depth_processed"
# csv_file = "/Volumes/Samsung USB/bcs_dict.csv"
#
# data_dict = {}
# label_dict = {}
#
# with open(csv_file, 'r') as csvfile:
#     reader = csv.reader(csvfile)
#     next(reader, None)
#     for row in reader:
#         key, value = row[0], row[1]
#         key_spl = key.split("_")
#         key_to_add = key_spl[0] + "_" + key_spl[1]
#         if key_to_add in label_dict:
#             raise Exception("Repeated keys")
#         label_dict[key_to_add] = value
#
# # get numpy arrays from root directory and add to data_dict
# filenames = os.listdir(root_dir)
# counter = 0
# for name in filenames:
#     if name[0] != ".":
#         video_np = np.load(os.path.join(root_dir, name), allow_pickle=True)
#         # print(type(video_np))
#         video = video_np['depth']
#         # print(type(video))
#         # val = np.array(video)
#         # print('type video:', type(video))
#         # print('type video[0]:', type(video[0]))
#         # print('type video[0][0]:', type(video[0][0]))
#         # print('type video[0][0][0]:', type(video[0][0][0]))
#
#         # print("vid shape", video.shape)
#
#         # if name == "120_20210119_adj10_floor5000.npz":
#         #     print(video)
#
#         # check types of every value
#         # for i in range(len(video)):
#         #     # if type(video[i]) != np.ndarray:
#         #     #     print(f"wrong type: {video[i]}, type: {type(video[i])}")
#         #     for j in range(len(video[i])):
#         #         # if type(video[i][j]) != np.ndarray:
#         #         #     print(f"wrong type: {video[i][j]}, type: {type(video[i][j])}")
#         #         for k in range(len(video[i][j])):
#         #             if type(video[i][j][k]) != np.float64:
#         #                 print(f"wrong type: {video[i][j][k]}, type: {type(video[i][j][k])}")
#
#         try:
#             tensor = torch.from_numpy(video)
#         except:
#             print(f"filename failure: {name}")
#             counter += 1
# print(cou)
#
#         # label = label_dict[search_name(name)]
#         # data_dict[video] = label
# #
# #
# # print(data_dict)
# # print(len(data_dict))
#
#
