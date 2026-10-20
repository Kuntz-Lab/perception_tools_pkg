#!/usr/bin/env python3
# SAM 2 video-tracking point cloud publisher
#
# Like segmented_pc_publisher's camera mode, but the bounding box only prompts the first frame: SAM 2's video
# memory then tracks the object wherever it moves in the image. Publishes the masked depth as a point cloud,
# plus an overlay image of the mask for debugging. Both carry the header stamp of the source RGB image, so
# latency is now - header.stamp (e.g. `ros2 topic delay /tissue_pointcloud`).
#   ros2 run perception_tools_pkg live_stream_video_pc_publisher [--sam-model small]
#
# To redraw the box while running (e.g. if tracking drifts):
#   ros2 service call /live_stream_video_pc_publisher/reprompt std_srvs/srv/Trigger

import argparse
import os
import sys
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_srvs.srv import Trigger

from perception_tools_pkg.utils.miscellaneous_utils import get_bb
from perception_tools_pkg.utils.ros_utils import numpy_to_pointcloud2
from perception_tools_pkg.utils.sam2_utils import add_sam2_args, resolve_sam2_model

OVERLAY_COLOR = (0, 255, 0)  # BGR


class LiveStreamVideoPointCloudPublisher(Node):
    def __init__(self,
                 sam_checkpoint_path,
                 sam_model_config,
                 cloud_name="tissue",
                 image_topic_name="/front/camera/color/image_raw",
                 depth_topic_name="/front/camera/aligned_depth_to_color/image_raw",
                 camera_info_topic_name="/front/camera/aligned_depth_to_color/camera_info",
                 publish_rate=100.0,
                 queue_size=10):
        super().__init__("live_stream_video_pc_publisher")

        self.get_logger().info(f"Using depth topic: {depth_topic_name}")

        # === Config & model setup ===
        if not os.path.isfile(sam_checkpoint_path):
            raise FileNotFoundError(f"SAM 2 checkpoint not found: {sam_checkpoint_path}")

        self.topic_name = f"/{cloud_name}_pointcloud"
        self.overlay_topic_name = f"/{cloud_name}_tracking_overlay"

        # SAM 2 tracker (imported here so the node's --help works without torch / sam2)
        import torch
        from perception_tools_pkg.sam2_stream_tracker import SAM2StreamTracker

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tracker = SAM2StreamTracker(sam_model_config, sam_checkpoint_path, device=device)
        self.get_logger().info(f"Loaded SAM 2 checkpoint {sam_checkpoint_path} on {device}")

        # Storage
        self.bridge = CvBridge()
        self.cur_img = None
        self.cur_depth = None
        self.cur_stamp = None
        self.has_new_frame = False
        self.frame_id = None
        self.camera_intrinsics = None
        self.reprompt_requested = True  # also asks for the initial box
        self.step_time = None  # smoothed SAM 2 time per frame (s)

        # ROS I/O
        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic_name, self.camera_info_callback, qos_profile_sensor_data)
        # RELIABLE, not sensor-data BEST_EFFORT: large image messages lose fragments in transit, and best effort
        # drops the whole frame (aligned depth arrived at ~0.2 Hz). A short history keeps frames from going stale.
        image_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=2, reliability=ReliabilityPolicy.RELIABLE)
        self.image_sub = Subscriber(self, Image, image_topic_name, qos_profile=image_qos)
        self.depth_sub = Subscriber(self, Image, depth_topic_name, qos_profile=image_qos)
        self.sync = ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], queue_size=queue_size, slop=0.05)
        self.sync.registerCallback(self.image_depth_callback)
        self.point_cloud_pub = self.create_publisher(PointCloud2, self.topic_name, queue_size)
        self.overlay_pub = self.create_publisher(Image, self.overlay_topic_name, queue_size)
        self.reprompt_srv = self.create_service(Trigger, "~/reprompt", self.reprompt_callback)

        # Timer for tracking and publishing; each tick processes the newest frame, skipping any that arrived
        # while SAM 2 was busy
        self.timer = self.create_timer(1.0 / publish_rate, self.track_and_publish)

        self.get_logger().info(
            f"LiveStreamVideoPointCloudPublisher initialized. Publishing to: {self.topic_name} and {self.overlay_topic_name}")

    # ===========================
    # CALLBACKS
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

    def image_depth_callback(self, image_msg, depth_msg):
        """Store a time-synchronized RGB image (RGB order, as SAM 2 expects) and depth image (uint16, mm)."""
        self.cur_stamp = image_msg.header.stamp
        self.frame_id = image_msg.header.frame_id
        self.cur_img = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="rgb8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        # cv_bridge tags the dtype with an explicit byte order ('<u2'), which Open3D's Image rejects; make it native
        self.cur_depth = depth.astype(depth.dtype.newbyteorder("="))
        self.has_new_frame = True

    def reprompt_callback(self, request, response):
        """Ask for a new bounding box on the next frame; tracking restarts from it."""
        self.reprompt_requested = True
        response.success = True
        response.message = "A bounding box window will open on the next frame."
        return response

    # ===========================
    # TRACK + PUBLISH
    # ===========================
    def track_and_publish(self):
        """Tracks the object in the newest frame with SAM 2 and publishes the point cloud and overlay."""
        import open3d as o3d

        if self.cur_img is None or self.cur_depth is None or self.camera_intrinsics is None:
            return

        t0 = time.perf_counter()
        if self.reprompt_requested:
            # Blocks this (single-threaded) node, so the frame boxed is the frame tracking starts from
            self.get_logger().info("Waiting for user to select bounding box...")
            box = get_bb(cv2.cvtColor(self.cur_img, cv2.COLOR_RGB2BGR), "Select the object to track")
            if box[2] <= box[0] or box[3] <= box[1]:
                # selectROI returns an empty box when cancelled (Esc / c); keep tracking if we were
                self.get_logger().warn("Empty bounding box" + (", keeping the current object."
                                                               if self.tracker.is_tracking else ", asking again."))
                self.reprompt_requested = not self.tracker.is_tracking
                return
            self.reprompt_requested = False
            self.get_logger().info(f"Bounding box selected: {box}")
            t0 = time.perf_counter()  # don't count the time spent drawing
            mask, object_score = self.tracker.start(self.cur_img, box)
        else:
            if not self.has_new_frame:
                return
            mask, object_score = self.tracker.track(self.cur_img)
        self.has_new_frame = False

        step_time = time.perf_counter() - t0
        self.step_time = step_time if self.step_time is None else 0.9 * self.step_time + 0.1 * step_time
        if object_score <= 0:
            self.get_logger().warn("Object not visible; publishing an empty point cloud.", throttle_duration_sec=2.0)

        # Zero out depth outside mask
        depth = np.array(self.cur_depth)
        depth[~mask] = 0

        # Construct point cloud
        pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth), self.camera_intrinsics)
        points = np.asarray(pcd.points)

        # Publish, stamped with the source image
        self.point_cloud_pub.publish(numpy_to_pointcloud2(points, frame_id=self.frame_id, stamp=self.cur_stamp))
        if self.overlay_pub.get_subscription_count() > 0:
            latency = (self.get_clock().now() - Time.from_msg(self.cur_stamp)).nanoseconds * 1e-9
            overlay_msg = self.bridge.cv2_to_imgmsg(
                self.make_overlay(self.cur_img, mask, object_score > 0, latency), encoding="bgr8")
            overlay_msg.header.stamp = self.cur_stamp
            overlay_msg.header.frame_id = self.frame_id
            self.overlay_pub.publish(overlay_msg)

    def make_overlay(self, image, mask, visible, latency):
        """BGR copy of the RGB image with the mask tinted and outlined, plus tracking status text."""
        overlay = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array(OVERLAY_COLOR)).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, OVERLAY_COLOR, 2)

        status = "tracking" if visible else "object not visible"
        text = f"{status} | SAM 2: {1000 * self.step_time:.0f} ms | latency: {1000 * latency:.0f} ms"
        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    OVERLAY_COLOR if visible else (0, 0, 255), 2, cv2.LINE_AA)
        return overlay


# ===========================
# MAIN
# ===========================
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Track an object with SAM 2 and publish its segmented point cloud.")
    parser.add_argument("--cloud-name", default="tissue",
                        help="Publishes to /<cloud-name>_pointcloud and /<cloud-name>_tracking_overlay.")
    parser.add_argument("--rate", type=float, default=100.0, help="Tracking timer rate in Hz (default: 100).")
    parser.add_argument("--image-topic", default="/front/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/front/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--camera-info-topic", default="/front/camera/aligned_depth_to_color/camera_info")
    add_sam2_args(parser)
    return parser.parse_args(argv)


def main(args=None):
    rclpy.init(args=args)
    cli_args = parse_args(remove_ros_args(args if args is not None else sys.argv)[1:])
    sam_checkpoint_path, sam_model_config = resolve_sam2_model(cli_args)

    node = LiveStreamVideoPointCloudPublisher(
        sam_checkpoint_path=sam_checkpoint_path,
        sam_model_config=sam_model_config,
        cloud_name=cli_args.cloud_name,
        image_topic_name=cli_args.image_topic,
        depth_topic_name=cli_args.depth_topic,
        camera_info_topic_name=cli_args.camera_info_topic,
        publish_rate=cli_args.rate,
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
