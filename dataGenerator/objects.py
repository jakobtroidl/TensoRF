import trimesh
import numpy as np
import torch
import os


import mesh2sdf


class Object3D:
    def __init__(self, name, vertices, faces, resolution):
        self.mesh = self.mesh = trimesh.Trimesh(vertices, faces)
        self.name = name
        self.res = resolution

    def __repr__(self):
        return f'Object3D(vertices={len(self.mesh.vertices)}, faces={len(self.mesh.faces)})'

    def normalize(self):
        # normalizes the object so that the coordinates
        # of all vertices are in the range [-1, 1]

        # Compute the centroid of the mesh
        centroid = self.mesh.centroid

        # Translate the mesh so that the centroid is at the origin
        self.mesh.vertices -= centroid

        # Compute the maximum absolute coordinate value
        max_abs_coord = np.max(np.abs(self.mesh.vertices))

        # Scale the mesh so that the coordinates of all vertices are in the range [-1, 1]
        self.mesh.vertices /= max_abs_coord

    def sample_sdf(self, as_tensor=True):
        # Compute the signed distance field of the mesh
        # @param size: the size of the SDF grid, result will be a size x size x size np array or tensor
        # @param as_tensor: whether to return the SDF as a numpy array or as a pytorch tensor
        # @return: the SDF grid as a pytorch tensor
        sdf = mesh2sdf.compute(self.mesh.vertices, self.mesh.faces, self.res)
        if as_tensor:
            sdf = torch.from_numpy(sdf)
        return sdf

    def store(self, sdf, path, as_tensor=True):
        parent = "{}/{}".format(path, self.name)
        if not os.path.exists(parent):
            # Create the directory
             os.makedirs(parent)
        if as_tensor:
            path = "{}/{}.pt".format(parent, self.res)
            torch.save(sdf, path)
        else:
            path = "{}/{}.npy".format(parent, self.res)
            with open(path, 'wb') as file:
                np.save(file, sdf)