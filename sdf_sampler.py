import argparse
import pywavefront
from dataGenerator.objects import Object3D

from params import Params


def main():
    parser = argparse.ArgumentParser(description='3D Object SDF Sampling Tool')
    parser.add_argument('--path', type=str, help='Path to the .obj file')
    parser.add_argument('--res', type=int, default=256, help='x, y, z resolution of the SDF grid')
    parser.add_argument('--out', type=str, default=Params.SDF_DIR, help='Output directory')
    args = parser.parse_args()

    # Load the .obj file
    scene = pywavefront.Wavefront(args.path, collect_faces=True)

    # Access the individual 3D objects
    for name, mesh in scene.meshes.items():
        print(f"Object name: {name}")
        obj = Object3D(name, scene.vertices, mesh.faces, args.res)
        sdf = obj.sample_sdf(args.out)
        obj.store(sdf, args.out)


if __name__ == '__main__':
    main()
