# Publishes the g30 tissue point cloud (file mode) plus the camera -> PSM1 calibration as a static transform.
#   ros2 launch perception_tools_pkg pc_with_camera_tf.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

# ===== Paste the output of interactive_tf_publisher over this block =====
# CAMERA_TF = {
#     'x': 0.000000, 'y': 0.278000, 'z': 0.000000,
#     'qx': 0.00000000, 'qy': 0.81411552, 'qz': -0.58070296, 'qw': -0.00000000,
#     'frame_id': 'PSM1_base_link',
#     'child_frame_id': 'camera_link',
# }

CAMERA_TF = {
    'x': -0.001000, 'y': 0.320000, 'z': 0.135000,
    'qx': 0.14338648, 'qy': 0.16643599, 'qz': -0.72396039, 'qw': 0.65392717,
    'frame_id': 'PSM1_base_link',
    'child_frame_id': 'camera_link',
}


def generate_launch_description():
    camera_tf_args = []
    for key, value in CAMERA_TF.items():
        camera_tf_args += [f'--{key.replace("_", "-")}', str(value)]

    return LaunchDescription([
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_tf_publisher',
            arguments=camera_tf_args,
        )
    ])
