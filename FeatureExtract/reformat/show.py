import numpy as np
from matplotlib import pyplot as plt

from utils.mvtec3d_util import *

tiff_path = f'datasets/mvtec3d/potato/train/good/xyz/293.tiff'

organized_pc = read_tiff_organized_pc(tiff_path)
depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
resized_organized_pc = resize_organized_pc(organized_pc)
depth = organized_pc[:, :, -1]
plt.imshow(depth)
plt.colorbar()
plt.show()
depth[depth == 0] = np.min(depth[depth != 0])
plt.imshow(depth)
plt.colorbar()
plt.show()
