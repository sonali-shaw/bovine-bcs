import numpy as np
import torch

nparr = np.array([[[1.0, 2.0, 3.0, 4.0 ], [1.0, 2.0, 3.0, 4.0]], [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]])
# print(nparr.shape())

# nparr = np.load("/Volumes/Samsung USB/depth_processed/120_20210119_adj10_floor5000.npz", allow_pickle=True)['depth']
# nparr2 = nparr.pad()
# print('type nparr',type(nparr))
# print('type nparr[0]',type(nparr[0]))
# print('type nparr:', type(nparr))
# print('type nparr[0]:', type(nparr[0]))
# print('type nparr[0][0]:', type(nparr[0][0]))
# print('type nparr[0][0][0]:', type(nparr[0][0][0]))
print(torch.from_numpy(nparr))