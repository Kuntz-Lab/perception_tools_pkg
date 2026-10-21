#!/usr/bin/env python3
# SAM 2 video-tracking point cloud publisher
#
# Like segmented_pc_publisher's camera mode, but each bounding box only prompts the first frame: SAM 2's video
# memory then tracks the object wherever it moves in the image. Tracks one object per point cloud topic and
# publishes each object's masked depth as a point cloud, plus one overlay image of all masks for debugging.
# All messages carry the header stamp of the source RGB image, so latency is now - header.stamp
# (e.g. `ros2 topic delay /tissue_pointcloud`).
#   ros2 run perception_tools_pkg live_stream_video_pc_publisher [--sam-model large]
#   ros2 run perception_tools_pkg live_stream_video_pc_publisher --cloud-topics /tissue_pointcloud /context_pointcloud
#
# To redraw the boxes while running (e.g. if tracking drifts), for all objects or one object:
#   ros2 service call /live_stream_video_pc_publisher/reprompt std_srvs/srv/Trigger
#   ros2 service call /live_stream_video_pc_publisher/reprompt/tissue std_srvs/srv/Trigger

import argparse
import os
import re
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

# One BGR color per object, in --cloud-topics order (repeats after the last)
OVERLAY_COLORS = [(0, 255, 0), (255, 0, 255), (255, 255, 0), (0, 165, 255), (0, 255, 255), (255, 128, 0)]
NOT_VISIBLE_COLOR = (0, 0, 255)


def object_label(cloud_topic):
    """Short object name for a point cloud topic, e.g. /front/tissue_pointcloud -> tissue."""
    name = cloud_topic.rstrip("/").split("/")[-1].removesuffix("_pointcloud")
    return re.sub(r"\W", "_", name)


class LiveStreamVideoPointCloudPublisher(Node):
    def __init__(self,
                 sam_checkpoint_path,
                 sam_model_config,
                 cloud_topics=("/tissue_pointcloud",),
                 overlay_topic_name="/tracking_overlay",
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

        # Tracked objects, keyed by label, in --cloud-topics order
        self.cloud_topics = {object_label(topic): topic for topic in cloud_topics}
        if len(self.cloud_topics) != len(cloud_topics):
            raise ValueError(f"Point cloud topics must have distinct names: {list(cloud_topics)}")
        self.labels = list(self.cloud_topics)
        self.colors = {label: OVERLAY_COLORS[i % len(OVERLAY_COLORS)] for i, label in enumerate(self.labels)}

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
        self.reprompt_labels = set(self.labels)  # objects waiting for a box, starting with all of them
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
        self.point_cloud_pubs = {label: self.create_publisher(PointCloud2, topic, queue_size)
                                 for label, topic in self.cloud_topics.items()}
        self.overlay_pub = self.create_publisher(Image, overlay_topic_name, queue_size)
        # ~/reprompt redraws every object's box; ~/reprompt/<label> redraws one
        self.reprompt_srvs = [self.create_service(
            Trigger, "~/reprompt", lambda req, res: self.reprompt_callback(self.labels, res))]
        for label in self.labels:
            self.reprompt_srvs.append(self.create_service(
                Trigger, f"~/reprompt/{label}", lambda req, res, label=label: self.reprompt_callback([label], res)))

        # Timer for tracking and publishing; each tick processes the newest frame, skipping any that arrived
        # while SAM 2 was busy
        self.timer = self.create_timer(1.0 / publish_rate, self.track_and_publish)

        self.get_logger().info(
            "LiveStreamVideoPointCloudPublisher initialized. Publishing to: "
            f"{', '.join(self.cloud_topics.values())} and {overlay_topic_name}")

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

    def reprompt_callback(self, labels, response):
        """Ask for new bounding boxes for labels on the next frame; their tracking restarts from them."""
        self.reprompt_labels.update(labels)
        response.success = True
        response.message = f"Bounding box windows for {', '.join(labels)} will open on the next frame."
        return response

    # ===========================
    # TRACK + PUBLISH
    # ===========================
    def select_boxes(self):
        """Ask the user for a box for each object waiting for one. Returns {label: box} of the boxes drawn."""
        # Blocks this (single-threaded) node, so the frame boxed is the frame tracking starts from
        image = cv2.cvtColor(self.cur_img, cv2.COLOR_RGB2BGR)
        boxes = {}
        for label in [label for label in self.labels if label in self.reprompt_labels]:
            self.get_logger().info(f"Waiting for user to select bounding box for {label}...")
            box = get_bb(image, f"Select the {label} object")
            if box[2] <= box[0] or box[3] <= box[1]:
                # selectROI returns an empty box when cancelled (Esc / c); keep tracking the object if we were
                self.get_logger().warn(f"Empty bounding box for {label}" + (
                    ", keeping the current object." if label in self.tracker.object_ids else ", asking again."))
                continue
            self.get_logger().info(f"Bounding box selected for {label}: {box}")
            boxes[label] = box
            # Show the boxes drawn so far while selecting the next one
            cv2.rectangle(image, tuple(map(int, box[:2])), tuple(map(int, box[2:])), self.colors[label], 2)
            cv2.putText(image, label, (int(box[0]), int(box[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        self.colors[label], 2, cv2.LINE_AA)
        # Objects without a box that aren't being tracked are asked for again on the next tick
        self.reprompt_labels = {label for label in self.reprompt_labels
                                if label not in boxes and label not in self.tracker.object_ids}
        return boxes

    def track_and_publish(self):
        """Tracks the objects in the newest frame with SAM 2 and publishes their point clouds and the overlay."""
        import open3d as o3d

        if self.cur_img is None or self.cur_depth is None or self.camera_intrinsics is None:
            return

        boxes = self.select_boxes() if self.reprompt_labels else {}
        if not boxes and (not self.has_new_frame or not self.tracker.object_ids):
            return
        self.has_new_frame = False

        t0 = time.perf_counter()
        results = self.tracker.track(self.cur_img, boxes)
        step_time = time.perf_counter() - t0
        self.step_time = step_time if self.step_time is None else 0.9 * self.step_time + 0.1 * step_time

        not_visible = [label for label, (_, object_score) in results.items() if object_score <= 0]
        if not_visible:
            self.get_logger().warn(f"Not visible, publishing empty point clouds: {', '.join(not_visible)}",
                                   throttle_duration_sec=2.0)

        # Publish one point cloud per object, stamped with the source image
        for label, (mask, _) in results.items():
            # Zero out depth outside mask
            depth = np.array(self.cur_depth)
            depth[~mask] = 0

            # Construct point cloud
            pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth), self.camera_intrinsics)
            points = np.asarray(pcd.points)
            self.point_cloud_pubs[label].publish(
                numpy_to_pointcloud2(points, frame_id=self.frame_id, stamp=self.cur_stamp))

        if self.overlay_pub.get_subscription_count() > 0:
            latency = (self.get_clock().now() - Time.from_msg(self.cur_stamp)).nanoseconds * 1e-9
            overlay_msg = self.bridge.cv2_to_imgmsg(self.make_overlay(self.cur_img, results, latency), encoding="bgr8")
            overlay_msg.header.stamp = self.cur_stamp
            overlay_msg.header.frame_id = self.frame_id
            self.overlay_pub.publish(overlay_msg)

    def make_overlay(self, image, results, latency):
        """BGR copy of the RGB image with each object's mask tinted and outlined, plus tracking status text."""
        overlay = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        lines = [(f"SAM 2: {1000 * self.step_time:.0f} ms | latency: {1000 * latency:.0f} ms", (255, 255, 255))]
        for label in self.labels:
            color = self.colors[label]
            if label not in results:
                lines.append((f"{label}: waiting for box", NOT_VISIBLE_COLOR))
                continue
            mask, object_score = results[label]
            lines.append((f"{label}: " + ("tracking" if object_score > 0 else "not visible"),
                          color if object_score > 0 else NOT_VISIBLE_COLOR))
            if not mask.any():
                continue
            overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array(color)).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)
            # Label above the top of the mask
            ys, xs = np.nonzero(mask)
            self.draw_text(overlay, label, (int(xs[ys.argmin()]), max(int(ys.min()) - 8, 20)), color)

        for i, (text, color) in enumerate(lines):
            self.draw_text(overlay, text, (10, 30 + 30 * i), color)
        return overlay

    @staticmethod
    def draw_text(image, text, origin, color):
        """Text with a dark outline, readable on any background."""
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


# ===========================
# MAIN
# ===========================
def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Track objects with SAM 2 and publish each object's segmented point cloud.")
    parser.add_argument("--cloud-topics", nargs="+", default=["/tissue_pointcloud"],
                        help="One point cloud topic per object to track (default: /tissue_pointcloud). Objects are "
                             "named after their topic, e.g. /context_pointcloud -> context.")
    parser.add_argument("--overlay-topic", default="/tracking_overlay",
                        help="Debug image of all objects' masks (default: /tracking_overlay).")
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
        cloud_topics=cli_args.cloud_topics,
        overlay_topic_name=cli_args.overlay_topic,
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
