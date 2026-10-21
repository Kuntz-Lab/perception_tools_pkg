# perception_tools_pkg

ROS 2 (Jazzy) perception tools for the dVRK.

## segmented_pc_publisher

Publishes a segmented point cloud on `/<cloud-name>_pointcloud` (default `/tissue_pointcloud`).
It has two modes:

- **camera** (default): subscribes to RGB, aligned depth, and camera info topics. You draw a bounding box once, SAM 2.1 segments every frame using that box, and the masked depth is published as a point cloud with the image's timestamp and frame.
- **file**: publishes an Nx3 `.npy` point cloud at a fixed rate, for testing without a camera. By default it publishes a real tissue cloud ([data/g30_tissue_cloud.npy](data/g30_tissue_cloud.npy)) in `camera_color_optical_frame`. That cloud is a snapshot of `/tissue_pointcloud` from the g30 recording. `data/` also has a matching `/context_pointcloud` snapshot (`g30_context_cloud.npy`) and a synthetic `dummy_cloud.npy`.

### Setup

Camera mode needs torch, SAM 2, and Open3D, which are installed with pip into a venv that can also see the system ROS packages:

```bash
python3 -m venv --system-site-packages ~/venvs/perception
source ~/venvs/perception/bin/activate
pip install -r requirements.txt
```

Download the SAM 2.1 checkpoints ([sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt), [sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)) to `~/model_weights/sam2/`. All SAM 2 nodes read them from there:

- `--sam-model small|large` picks the model (default `small`).
- To keep the weights somewhere else, change `SAM2_WEIGHTS_DIR` in [perception_tools_pkg/utils/sam2_utils.py](perception_tools_pkg/utils/sam2_utils.py) or set the `SAM2_WEIGHTS_DIR` environment variable.
- `--sam-checkpoint` / `--sam-config` override the file for a single run.

Build with the venv active:

```bash
cd ~/dvrk_ws
colcon build --packages-select perception_tools_pkg
source install/setup.bash
```

#### Recommended: large-data transport

Set this in every terminal that runs ROS nodes, or add it to `~/.bashrc`:

```bash
export FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA
```

Restart the camera driver, RViz and these nodes after setting it.

**Why:** these nodes subscribe to full-resolution camera images (2.7 MB color and 1.8 MB depth per frame at 30 Hz) and publish point clouds and a debug image. By default, Fast DDS (the default DDS in ROS 2 Jazzy) sends each large message as many UDP fragments through small kernel socket buffers (208 KB by default). When fragments are dropped, reliable topics resend them. That repair traffic is shared by every subscriber of a publisher, so adding subscribers or large topics makes all image streams lag, including ones these nodes don't use.

In practice, viewing `/tracking_overlay` in RViz alongside the camera image made the plain image stream lag, and new subscribers such as `ros2 topic hz` received no frames at all. `LARGE_DATA` sends large messages over shared memory (same machine) and TCP instead, which fixed it. The variable must be set for both the publisher and the subscriber of a topic.

### Running

Always activate the venv first. The node runs with whichever `python3` is first on `PATH`, and an active conda base env will otherwise take precedence.

```bash
source ~/venvs/perception/bin/activate

# Camera mode
ros2 run perception_tools_pkg segmented_pc_publisher

# File mode (g30 tissue cloud; use --file for another .npy)
ros2 run perception_tools_pkg segmented_pc_publisher --mode file
```

Run with `--help` to see all options (topics, `--cloud-name`, `--rate`, `--file`, `--frame-id`, ...).

When testing file mode without a camera driver, nothing broadcasts `camera_color_optical_frame` on TF. RViz still draws the cloud if its Fixed Frame is typed in manually, but it reports that the frame does not exist. To fix this, publish a static transform:

```bash
ros2 run tf2_ros static_transform_publisher --frame-id world --child-frame-id camera_color_optical_frame
```

To regenerate the dummy cloud, run `python3 scripts/make_dummy_cloud.py`.

## live_stream_video_pc_publisher

Tracks objects with SAM 2 video segmentation, so each one stays segmented wherever it moves in the frame. At startup you draw one bounding box per object. It uses the same camera topics and setup as `segmented_pc_publisher` camera mode.

```bash
source ~/venvs/perception/bin/activate
ros2 run perception_tools_pkg live_stream_video_pc_publisher  # one object on /tissue_pointcloud, SAM 2.1 Small
ros2 run perception_tools_pkg live_stream_video_pc_publisher --sam-model large  # more accurate, slower

# One object per point cloud topic, named after the topic (tissue, context)
ros2 run perception_tools_pkg live_stream_video_pc_publisher --cloud-topics /tissue_pointcloud /context_pointcloud

# Redraw the boxes mid-stream (tracking restarts from the new boxes): all objects, or one by name
ros2 service call /live_stream_video_pc_publisher/reprompt std_srvs/srv/Trigger
ros2 service call /live_stream_video_pc_publisher/reprompt/context std_srvs/srv/Trigger
```

It publishes:

- one point cloud per object, on the topics given by `--cloud-topics`. A cloud is empty while its object is out of view.
- `/tracking_overlay`: the image with every object's mask drawn on it, for debugging.

All messages carry the source RGB image's stamp, so `ros2 topic delay /tissue_pointcloud` shows the latency.

Track several objects in one node rather than running several nodes, since the image encoder then runs once per frame for all objects. On an RTX 5090, SAM 2 takes ~29 ms per frame with Large and ~13 ms with Small for one object, plus ~7 ms for each additional object. Use `--help` for all options.

## Camera to PSM1 calibration

`interactive_tf_publisher` publishes a static transform from `PSM1_base_link` (parent) to `camera_color_optical_frame` (child). It starts at identity, and you adjust it from the keyboard while watching the result in RViz:

```bash
ros2 run perception_tools_pkg interactive_tf_publisher   # press h for the key list
```

- Translation keys move along the parent's axes, and rotation keys rotate about the child's own axes.
- Use `--parent-frame` and `--child-frame` to calibrate different frames.
- Use `--x ... --qw` to start from an earlier result.
- Nothing else may publish a parent for the child frame. For example, run the RealSense driver with `publish_tf:=false`.

When you press Ctrl-C, the node prints a `CAMERA_TF = {...}` block. Paste it over the block in [launch/pc_with_camera_tf.launch.py](launch/pc_with_camera_tf.launch.py) and rebuild, unless you built with `--symlink-install`. The launch file then publishes that transform with the standard `static_transform_publisher`, along with the point cloud publisher in file mode:

```bash
ros2 launch perception_tools_pkg pc_with_camera_tf.launch.py
```
