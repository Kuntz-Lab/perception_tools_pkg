# Publishes the g30 tissue point cloud (file mode) plus the camera -> PSM1 calibration as a static transform.
#   ros2 launch perception_tools_pkg pc_with_camera_tf.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node

# ===== Paste the output of interactive_tf_publisher over this block =====
CAMERA_TF = {
    'x': 0.000000, 'y': 0.278000, 'z': 0.000000,
    'qx': 0.00000000, 'qy': 0.81411552, 'qz': -0.58070296, 'qw': -0.00000000,
    'frame_id': 'PSM1_base_link',
    'child_frame_id': 'camera_color_optical_frame',
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
        ),
        Node(
            package='perception_tools_pkg',
            executable='segmented_pc_publisher',
            arguments=['--mode', 'file'],
            output='screen',
        ),
    ])
