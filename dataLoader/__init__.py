from .llff import LLFFDataset
from .blender import BlenderDataset
from .nsvf import NSVF
from .tankstemple import TanksTempleDataset
from .your_own_data import YourOwnDataset
from .sdf_data import RegularSDFDataset



dataset_dict = {'blender': BlenderDataset,
               'llff':LLFFDataset,
               'tankstemple':TanksTempleDataset,
               'nsvf':NSVF,
                'reg_sdf':RegularSDFDataset,
                'own_data':YourOwnDataset}