#!/usr/bin/env python3
# Bounding-box-prompted SAM 2 point cloud publisher (ROS 2 port of dvrk_python pc_publisher.py)
#
# Camera mode (default): segments the RGB image with SAM 2 using a bounding box selected once
# by the user, masks the aligned depth image, and publishes the resulting point cloud.
#   ros2 run perception_tools_pkg segmented_pc_publisher --sam-checkpoint /path/to/sam2.1_hiera_large.pt
#
# File mode: publishes an Nx3 point cloud loaded from a .npy file, for testing without a camera.
#   ros2 run perception_tools_pkg segmented_pc_publisher --mode file

import argparse
import os
import sys

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo, Image, PointCloud2

from perception_tools_pkg.utils.miscellaneous_utils import get_bb
from perception_tools_pkg.utils.ros_utils import numpy_to_pointcloud2

# Path to the SAM 2.1 Large checkpoint (sam2.1_hiera_large.pt). Left blank until the weights are downloaded;
# can also be passed with --sam-checkpoint.
SAM2_CHECKPOINT_PATH = ""
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


class PointCloudPublisher(Node):
    def __init__(self,
                 cloud_name="tissue",
                 image_topic_name="/camera/camera/color/image_raw",
                 depth_topic_name="/camera/camera/aligned_depth_to_color/image_raw",
                 camera_info_topic_name="/camera/camera/aligned_depth_to_color/camera_info",
                 sam_checkpoint_path=SAM2_CHECKPOINT_PATH,
                 sam_model_config=SAM2_MODEL_CONFIG,
                 publish_rate=100.0,
                 queue_size=10):
        super().__init__("pc_publisher")

        self.get_logger().info(f"Using depth topic: {depth_topic_name}")

        # === Config & model setup ===
        if not sam_checkpoint_path:
            raise ValueError("No SAM 2 checkpoint given. Set SAM2_CHECKPOINT_PATH or pass --sam-checkpoint.")
        if not os.path.isfile(sam_checkpoint_path):
            raise FileNotFoundError(f"SAM 2 checkpoint not found: {sam_checkpoint_path}")

        self.topic_name = f"/{cloud_name}_pointcloud"

        # SAM 2 model (imported here so file mode does not need torch / sam2 installed)
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_sam2(sam_model_config, sam_checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)

        # Storage
        self.bridge = CvBridge()
        self.cur_img = None
        self.cur_depth = None
        self.cur_stamp = None
        self.has_new_frame = False
        self.bounding_box = None
        self.frame_id = None
        self.camera_intrinsics = None

        # ROS I/O
        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic_name, self.camera_info_callback, qos_profile_sensor_data)
        self.image_sub = Subscriber(self, Image, image_topic_name, qos_profile=qos_profile_sensor_data)
        self.depth_sub = Subscriber(self, Image, depth_topic_name, qos_profile=qos_profile_sensor_data)
        self.sync = ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], queue_size=queue_size, slop=0.05)
        self.sync.registerCallback(self.image_depth_callback)
        self.point_cloud_pub = self.create_publisher(PointCloud2, self.topic_name, queue_size)

        # Timer for publishing
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_point_cloud)

        self.get_logger().info(f"PointCloudPublisher initialized. Publishing to: {self.topic_name}")

    # ===========================
    # CAMERA INFO CALLBACK
    # ===========================
    def camera_info_callback(self, msg):
        import open3d as o3d

        if self.camera_intrinsics is not None:
            return  # Already received

        intrinsic_matrix = np.array(msg.k).reshape(3, 3)

        self.camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=msg.width,
            height=msg.height,
            fx=intrinsic_matrix[0, 0],
            fy=intrinsic_matrix[1, 1],
            cx=intrinsic_matrix[0, 2],
            cy=intrinsic_matrix[1, 2]
        )

        self.get_logger().info("Camera intrinsics received.")

    # ===========================
    # IMAGE + DEPTH CALLBACK
    # ===========================
    def image_depth_callback(self, image_msg, depth_msg):
        """Store a time-synchronized RGB image (RGB order, as SAM 2 expects) and depth image (uint16, mm)."""
        self.cur_stamp = image_msg.header.stamp
        self.frame_id = image_msg.header.frame_id
        self.cur_img = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="rgb8")
        self.cur_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        self.has_new_frame = True

    # ===========================
    # PUBLISH POINT CLOUD
    # ===========================
    def publish_point_cloud(self):
        """Runs SAM 2 segmentation using the pre-selected bounding box and publishes the point cloud."""
        import open3d as o3d

        if self.cur_img is None or self.cur_depth is None or self.camera_intrinsics is None:
            return

        # Prompt user for bounding box ONCE
        if self.bounding_box is None:
            self.get_logger().info("Waiting for user to select bounding box...")
            self.bounding_box = get_bb(cv2.cvtColor(self.cur_img, cv2.COLOR_RGB2BGR), "Select the bounding box")
            self.get_logger().info(f"Bounding box selected: {self.bounding_box}")

        if not self.has_new_frame:
            return
        self.has_new_frame = False

        # Run SAM 2 segmentation using *only* bounding box
        with self.torch.inference_mode(), self.torch.autocast(
                self.device.type, dtype=self.torch.bfloat16, enabled=self.device.type == "cuda"):
            self.predictor.set_image(self.cur_img)
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=self.bounding_box[None, :],
                multimask_output=False
            )

        best_mask = masks[np.argmax(scores)].astype(bool)

        # Zero out depth outside mask
        depth = np.array(self.cur_depth)
        depth[~best_mask] = 0

        # Construct point cloud
        pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth), self.camera_intrinsics)
        points = np.asarray(pcd.points)

        # Publish
        msg = numpy_to_pointcloud2(points, frame_id=self.frame_id, stamp=self.cur_stamp)
        self.point_cloud_pub.publish(msg)


class FilePointCloudPublisher(Node):
    """Publishes a fixed Nx3 point cloud loaded from a .npy file, for testing downstream consumers."""

    def __init__(self, cloud_file, cloud_name="tissue", frame_id="camera_color_optical_frame", publish_rate=10.0, queue_size=10):
        super().__init__("pc_publisher")

        self.topic_name = f"/{cloud_name}_pointcloud"
        self.frame_id = frame_id
        self.points = np.load(cloud_file)
        if self.points.ndim != 2 or self.points.shape[1] < 3:
            raise ValueError(f"Expected an Nx3 array in {cloud_file}, got shape {self.points.shape}")

        self.point_cloud_pub = self.create_publisher(PointCloud2, self.topic_name, queue_size)
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_point_cloud)

        self.get_logger().info(
            f"Loaded {self.points.shape[0]} points from {cloud_file}. Publishing to: {self.topic_name}")

    def publish_point_cloud(self):
        msg = numpy_to_pointcloud2(self.points, frame_id=self.frame_id, stamp=self.get_clock().now().to_msg())
        self.point_cloud_pub.publish(msg)


# ===========================
# MAIN
# ===========================
def parse_args(argv):
    default_cloud_file = os.path.join(get_package_share_directory("perception_tools_pkg"), "data", "g30_tissue_cloud.npy")

    parser = argparse.ArgumentParser(description="Publish a SAM 2 segmented point cloud, or one loaded from file.")
    parser.add_argument("--mode", choices=["camera", "file"], default="camera",
                        help="camera: segment live image + depth topics. file: publish a .npy point cloud.")
    parser.add_argument("--cloud-name", default="tissue", help="Publishes to /<cloud-name>_pointcloud.")
    parser.add_argument("--rate", type=float, default=None,
                        help="Publish timer rate in Hz (default: 100 for camera, 10 for file).")

    camera = parser.add_argument_group("camera mode")
    camera.add_argument("--image-topic", default="/camera/camera/color/image_raw")
    camera.add_argument("--depth-topic", default="/camera/camera/aligned_depth_to_color/image_raw")
    camera.add_argument("--camera-info-topic", default="/camera/camera/aligned_depth_to_color/camera_info")
    camera.add_argument("--sam-checkpoint", default=SAM2_CHECKPOINT_PATH)
    camera.add_argument("--sam-config", default=SAM2_MODEL_CONFIG)

    file_group = parser.add_argument_group("file mode")
    file_group.add_argument("--file", default=default_cloud_file, help="Nx3 .npy point cloud (default: g30 tissue cloud).")
    file_group.add_argument("--frame-id", default="camera_color_optical_frame",
                            help="Frame of the loaded cloud (default matches the RealSense color optical frame).")

    return parser.parse_args(argv)


def main(args=None):
    rclpy.init(args=args)
    cli_args = parse_args(remove_ros_args(args if args is not None else sys.argv)[1:])

    if cli_args.mode == "camera":
        node = PointCloudPublisher(
            cloud_name=cli_args.cloud_name,
            image_topic_name=cli_args.image_topic,
            depth_topic_name=cli_args.depth_topic,
            camera_info_topic_name=cli_args.camera_info_topic,
            sam_checkpoint_path=cli_args.sam_checkpoint,
            sam_model_config=cli_args.sam_config,
            publish_rate=cli_args.rate or 100.0,
        )
    else:
        node = FilePointCloudPublisher(
            cloud_file=cli_args.file,
            cloud_name=cli_args.cloud_name,
            frame_id=cli_args.frame_id,
            publish_rate=cli_args.rate or 10.0,
        )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
