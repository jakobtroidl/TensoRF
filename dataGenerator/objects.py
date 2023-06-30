import trimesh
import numpy as np
import torch
import os
import torchvision.utils as vutils

import mesh2sdf


class Object3D:
    def __init__(self, name, vertices, faces, resolution):
        self.mesh = self.mesh = trimesh.Trimesh(vertices, faces)
        self.name = name
        self.res = resolution
        self.level = 2 / self.res
        self.mesh_scale = 0.8
        self.normalize()

    def __repr__(self):
        return f'Object3D(vertices={len(self.mesh.vertices)}, faces={len(self.mesh.faces)})'

    def normalize(self):
        vertices = self.mesh.vertices
        bbmin = vertices.min(0)
        bbmax = vertices.max(0)
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 * self.mesh_scale / (bbmax - bbmin).max()
        self.mesh.vertices = (vertices - center) * scale

    def sample_sdf(self, as_tensor=True):
        # Compute the signed distance field of the mesh
        # @param size: the size of the SDF grid, result will be a size x size x size np array or tensor
        # @param as_tensor: whether to return the SDF as a numpy array or as a pytorch tensor
        # @return: the SDF grid as a pytorch tensor
        sdf = mesh2sdf.compute(self.mesh.vertices, self.mesh.faces, self.res, fix=True, level=self.level, return_mesh=False)
        if as_tensor:
            sdf = torch.from_numpy(sdf)
        return sdf
    
    def save_as_images(self, sdf):
        # Save the SDF as a set of 2D images
        # @param sdf: the SDF grid as a pytorch tensor
        # @return: None

        # Create the directory
        path = 'log/{}/imgs_vis'.format('bunny_sdf')
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save the images
        for i in range(self.res):
            img = sdf[:, :, i]
            vutils.save_image(img, path + "/" + f'{i}.png', normalize=True)


    def store(self, sdf, path, as_tensor=True):
        parent = "{}/{}".format(path, self.name)
        if not os.path.exists(parent):
            # Create the directory
             os.makedirs(parent)
        if as_tensor:
            path = "{}/{}_{}_{}.pt".format(parent, self.res, self.res, self.res)
            torch.save(sdf, path)
        else:
            path = "{}/{}_{}_{}.npy".format(parent, self.res, self.res, self.res)
            with open(path, 'wb') as file:
                np.save(file, sdf)
