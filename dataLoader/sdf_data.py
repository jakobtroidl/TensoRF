import torch, cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *


class RegularSDFDataset(Dataset):
    def __init__(self, datadir, res=[256, 256, 256], downsample=1.0):

        self.root_dir = datadir
        self.res = res
        self.all_pos = None
        self.all_sdf = None
        self.near_far = [0, res]
        self.white_bg = False

        self.scene_bbox = self.get_aabb(res)
        self.read_meta()

    # def read_depth(self, filename):
    #     depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
    #     return depth

    @staticmethod
    def get_aabb(res):
        return torch.tensor([[0, 0, 0], res])

    def read_meta(self):

        path = "{}/{}_{}_{}.pt".format(self.root_dir, self.res[0], self.res[1], self.res[2])

        # load preprocessed data
        if os.path.exists(path):
            data = torch.load(path)
        else:
            # throw exception if no preprocessed data
            raise Exception("No preprocessed data found at {}.".format(path))

        # get size of data as array
        # Create index tensors for each dimension
        x_idx = torch.linspace(-0.5, 0.5, self.res[0])
        y_idx = torch.linspace(-0.5, 0.5, self.res[1])
        z_idx = torch.linspace(-0.5, 0.5, self.res[2])

        # Create 3D meshgrid
        meshgrid = torch.meshgrid(x_idx, y_idx, z_idx)

        x, y, z, = meshgrid

        x_flat = torch.flatten(x)
        y_flat = torch.flatten(y)
        z_flat = torch.flatten(z)

        pos = torch.stack([x_flat, y_flat, z_flat], dim=1)
        sdf = torch.flatten(data)

        # concat into 2D tensor
        self.all_pos = pos.float()

        # get the SDF values
        self.all_sdf = sdf.float()

        print("Max {}".format(torch.max(self.all_sdf)))
        print("Min {}".format(torch.min(self.all_sdf)))    
              

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
