import pywavefront
from objects import Object3D

filename = '../data/models/test.obj'

# Load the .obj file
scene = pywavefront.Wavefront(filename, collect_faces=True)

# Access the individual 3D objects
for name, mesh in scene.meshes.items():
    print(f"Object name: {name}")
    obj = Object3D(name, scene.vertices, mesh.faces)
    sdf = obj.sample_sdf(cache=True)










