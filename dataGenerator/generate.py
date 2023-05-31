import argparse
import pywavefront

import params
from objects import Object3D


def main():
    parser = argparse.ArgumentParser(description='3D Object SDF Sampling Tool')
    parser.add_argument('--in', type=str, help='Path to the .obj file')
    parser.add_argument('--res', type=int, default=256, help='Number of SDF samples per object')
    parser.add_argument('--out', type=str, default=params.sdf_dir, help='Output directory')
    args = parser.parse_args()

    # Load the .obj file
    scene = pywavefront.Wavefront(args.filename, collect_faces=True)

    # Access the individual 3D objects
    for name, mesh in scene.meshes.items():
        print(f"Object name: {name}")
        obj = Object3D(name, scene.vertices, mesh.faces, args.r)
        sdf = obj.sample_sdf(args.o)
        obj.store(sdf, args.out)


if __name__ == '__main__':
    main()
