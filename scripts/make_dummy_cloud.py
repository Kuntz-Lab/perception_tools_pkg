#!/usr/bin/env python3
"""Generate a dummy Nx3 point cloud (a wavy 10 cm x 10 cm surface patch, ~30 cm from the camera) for file mode."""

import argparse
import os

import numpy as np


def make_dummy_cloud(num_points_per_side=50, size=0.1, depth=0.3, amplitude=0.01):
    xs = np.linspace(-size / 2, size / 2, num_points_per_side)
    x, y = np.meshgrid(xs, xs)
    z = depth + amplitude * np.sin(2 * np.pi * x / size) * np.cos(2 * np.pi * y / size)
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(np.float32)


if __name__ == "__main__":
    default_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "dummy_cloud.npy")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=default_output)
    args = parser.parse_args()

    cloud = make_dummy_cloud()
    np.save(args.output, cloud)
    print(f"Saved {cloud.shape} point cloud to {os.path.normpath(args.output)}")
