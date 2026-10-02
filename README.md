# perception_tools_pkg

ROS 2 (Jazzy) perception tools for the dVRK.

## segmented_pc_publisher

Publishes a segmented point cloud on `/<cloud-name>_pointcloud` (default `/tissue_pointcloud`).
It has two modes:

- **camera** (default): subscribes to RGB, aligned depth, and camera info topics. You draw a bounding box once, SAM 2.1 segments every frame using that box, and the masked depth is published as a point cloud with the image's timestamp and frame.
- **file**: publishes an Nx3 `.npy` point cloud at a fixed rate, for testing without a camera. By default it publishes the bundled dummy cloud ([data/dummy_cloud.npy](data/dummy_cloud.npy)) in `camera_color_optical_frame`.

### Setup

Camera mode needs torch, SAM 2, and Open3D, which are installed with pip into a venv that can also see the system ROS packages:

```bash
python3 -m venv --system-site-packages ~/venvs/perception
source ~/venvs/perception/bin/activate
pip install -r requirements.txt
```

Download the SAM 2.1 Large checkpoint ([sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)), e.g. to `~/weight_downloads/sam2/`. Then either set `SAM2_CHECKPOINT_PATH` in the node or pass `--sam-checkpoint`.

Build with the venv active:

```bash
cd ~/dvrk_ws
colcon build --packages-select perception_tools_pkg
source install/setup.bash
```

### Running

Always activate the venv first. The node runs with whichever `python3` is first on `PATH`, and an active conda base env will otherwise take precedence.

```bash
source ~/venvs/perception/bin/activate

# Camera mode
ros2 run perception_tools_pkg segmented_pc_publisher --sam-checkpoint ~/weight_downloads/sam2/sam2.1_hiera_large.pt

# File mode (dummy cloud)
ros2 run perception_tools_pkg segmented_pc_publisher --mode file
```

Run with `--help` to see all options (topics, `--cloud-name`, `--rate`, `--file`, `--frame-id`, ...).

When testing file mode without a camera driver, nothing broadcasts `camera_color_optical_frame` on TF. RViz still draws the cloud if its Fixed Frame is typed in manually, but it reports that the frame does not exist. To fix this, publish a static transform:

```bash
ros2 run tf2_ros static_transform_publisher --frame-id world --child-frame-id camera_color_optical_frame
```

To regenerate the dummy cloud, run `python3 scripts/make_dummy_cloud.py`.
