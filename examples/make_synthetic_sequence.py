"""Generate small synthetic PHDF inputs; these are not physical GRMHD simulations."""

import argparse
from pathlib import Path

import h5py
import numpy as np


def make_sequence(output_dir, frames=51):
    output_dir = Path(output_dir)
    if frames < 3:
        raise ValueError("At least three states are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("Use an empty directory for synthetic inputs")
    phi, theta, radius = np.meshgrid(
        np.linspace(0, 2 * np.pi, 8), np.linspace(-1, 1, 16),
        np.linspace(0, 1, 64), indexing="ij",
    )
    for sequence in range(frames):
        phase = 0.17 * sequence
        density = np.exp(-2 * radius) * (1 + 0.1 * np.sin(phi + phase))
        internal = density * np.exp(-1 + 0.2 * np.cos(theta - phase))
        velocity = np.stack([0.1 * np.sin(phi + phase), 0.1 * np.cos(theta + phase),
                             0.03 * np.cos(radius - phase)])
        magnetic = np.stack([0.02 * np.cos(phi - phase), 0.03 * np.sin(theta + phase),
                             0.04 * np.cos(radius + phase)])
        arrays = {"prims.rho": density[None], "prims.u": internal[None],
                  "prims.uvec": velocity[None], "prims.B": magnetic[None]}
        path = output_dir / f"synthetic.out0.{sequence:05d}.phdf"
        with h5py.File(path, "w") as handle:
            info = handle.create_group("Info")
            info.attrs["Time"] = 0.1 * sequence + 0.001 * np.sin(sequence)
            info.attrs["NumMeshBlocks"] = 1
            info.attrs["MeshBlockSize"] = [64, 16, 8]
            info.attrs["IncludesGhost"] = 0
            info.attrs["Multilevel"] = 0
            handle.create_group("Input").attrs["File"] = (
                "<parthenon/mesh>\nnx1=64\nnx2=16\nnx3=8\nx1min=0\nx1max=6\n"
                "x2min=0\nx2max=1\nx3min=0\nx3max=6.283185307179586\n"
                "<coordinates>\na=0.5\nr_in=1\nr_out=403.428793\ntransform=fmks\n"
                "hslope=0.3\nmks_smooth=0.5\npoly_xt=0.82\npoly_alpha=14\n"
                "<GRMHD>\ngamma=1.3333333333333333\n"
            )
            handle.create_group("Blocks")["loc.lx123"] = np.array([[0, 0, 0]], dtype="i8")
            for name, values in arrays.items():
                handle.create_dataset(name, data=values.astype("f4"), compression="gzip",
                                      compression_opts=1, shuffle=True)
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=51)
    arguments = parser.parse_args()
    print(make_sequence(arguments.output_dir, arguments.frames))
