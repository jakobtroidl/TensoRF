import torch, cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *


class RegularSDFDataset(Dataset):
    def __init__(self, datadir, res=256, downsample=1.0):
        self.root_dir = datadir
        self.res = res
        self.all_pos = None
        self.all_sdf = None
        self.read_meta()

    # def read_depth(self, filename):
    #     depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
    #     return depth

    def read_meta(self):

        path = "{}/{}.pt".format(self.root_dir, self.res)

        # load preprocessed data
        if os.path.exists(path):
            data = torch.load(path)
        else:
            # throw exception if no preprocessed data
            raise Exception("No preprocessed data found at {}.".format(path))

        # get size of data as array
        # Create index tensors for each dimension
        idx = torch.arange(self.res)

        # Create 3D meshgrid
        meshgrid = torch.meshgrid(idx, idx, idx)

        x, y, z, = meshgrid

        x_flat = torch.flatten(x)
        y_flat = torch.flatten(y)
        z_flat = torch.flatten(z)

        # concat into 2D tensor
        self.all_pos = torch.stack([x_flat, y_flat, z_flat], dim=1)

        # get the SDF values
        self.all_sdf = torch.flatten(data)

    # def define_transforms(self):
    #     self.transform = T.ToTensor()
    #
    # def define_proj_mat(self):
    #     self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:, :3]
    #
    # def world2ndc(self, points, lindisp=None):
    #     device = points.device
    #     return (points - self.center.to(device)) / self.radius.to(device)

    def __len__(self):
        return len(self.all_pos)

    def __getitem__(self, idx):
        pos = self.all_pos[idx]
        sdf = self.all_sdf[idx]

        sample = {'position': pos, 'sdf': sdf}

        return sample
