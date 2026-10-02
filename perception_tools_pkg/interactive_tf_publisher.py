#!/usr/bin/env python3
# Interactive static transform publisher for hand-tuning a frame (e.g. the camera) relative to another
# (e.g. the PSM base) while watching the result in RViz.
#
# Starts at identity (or at --x ... --qw if given) and nudges the transform with single keypresses.
# On exit (Ctrl-C), prints the final transform as a block to paste into launch/pc_with_camera_tf.launch.py.
#   ros2 run perception_tools_pkg interactive_tf_publisher

import argparse
import os
import select
import signal
import sys
import termios
import tty

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args
from scipy.spatial.transform import Rotation
from tf2_msgs.msg import TFMessage

TRANSLATION_STEPS = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]  # m
ROTATION_STEPS = [0.1, 0.5, 1.0, 5.0, 15.0]  # deg

# key: (translation axis or None, rotation axis or None, sign)
MOVE_KEYS = {
    "q": (0, None, +1), "a": (0, None, -1),
    "w": (1, None, +1), "s": (1, None, -1),
    "e": (2, None, +1), "d": (2, None, -1),
    "u": (None, 0, +1), "j": (None, 0, -1),
    "i": (None, 1, +1), "k": (None, 1, -1),
    "o": (None, 2, +1), "l": (None, 2, -1),
}

HELP = """
Interactive TF publisher: {parent} -> {child}

  Translate (along {parent} axes)        Rotate (about {child}'s own axes)
    q / a : +x / -x                          u / j : +roll  / -roll   (about x)
    w / s : +y / -y                          i / k : +pitch / -pitch  (about y)
    e / d : +z / -z                          o / l : +yaw   / -yaw    (about z)

  = / -  : bigger / smaller translation step
  ] / [  : bigger / smaller rotation step
  p      : print current transform
  r      : reset to identity
  h      : show this help
  Ctrl-C : quit and print the final transform
"""


class InteractiveTfPublisher(Node):
    def __init__(self, parent_frame, child_frame, translation, quaternion):
        super().__init__("interactive_tf_publisher")

        self.parent_frame = parent_frame
        self.child_frame = child_frame
        self.translation = np.array(translation, dtype=float)
        self.rotation = Rotation.from_quat(quaternion)  # scipy order: x, y, z, w

        self.translation_step_idx = TRANSLATION_STEPS.index(0.001)
        self.rotation_step_idx = ROTATION_STEPS.index(1.0)

        # Publish to /tf_static directly: tf2_ros.StaticTransformBroadcaster ignores updates to an existing
        # child frame, so it would keep sending the first (identity) transform
        tf_static_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                   history=HistoryPolicy.KEEP_LAST)
        self.tf_static_pub = self.create_publisher(TFMessage, "/tf_static", tf_static_qos)
        self.publish_transform()

    @property
    def translation_step(self):
        return TRANSLATION_STEPS[self.translation_step_idx]

    @property
    def rotation_step(self):
        return ROTATION_STEPS[self.rotation_step_idx]

    def publish_transform(self):
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.parent_frame
        msg.child_frame_id = self.child_frame
        msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = self.translation
        qx, qy, qz, qw = self.rotation.as_quat()
        msg.transform.rotation.x = qx
        msg.transform.rotation.y = qy
        msg.transform.rotation.z = qz
        msg.transform.rotation.w = qw
        self.tf_static_pub.publish(TFMessage(transforms=[msg]))

    def handle_key(self, key):
        """Apply one keypress. Returns a status line to show, or None."""
        if not key:
            return None
        if key in MOVE_KEYS:
            translation_axis, rotation_axis, sign = MOVE_KEYS[key]
            if translation_axis is not None:
                self.translation[translation_axis] += sign * self.translation_step
            else:
                # Right-multiply so the rotation is about the child frame's own axes (child origin stays put)
                rotvec = np.zeros(3)
                rotvec[rotation_axis] = sign * np.deg2rad(self.rotation_step)
                self.rotation = self.rotation * Rotation.from_rotvec(rotvec)
            self.publish_transform()
            return self.status_line()
        if key in "=+":
            self.translation_step_idx = min(self.translation_step_idx + 1, len(TRANSLATION_STEPS) - 1)
            return self.status_line()
        if key in "-_":
            self.translation_step_idx = max(self.translation_step_idx - 1, 0)
            return self.status_line()
        if key == "]":
            self.rotation_step_idx = min(self.rotation_step_idx + 1, len(ROTATION_STEPS) - 1)
            return self.status_line()
        if key == "[":
            self.rotation_step_idx = max(self.rotation_step_idx - 1, 0)
            return self.status_line()
        if key == "r":
            self.translation = np.zeros(3)
            self.rotation = Rotation.identity()
            self.publish_transform()
            return self.status_line()
        if key == "p":
            return self.launch_block()
        if key == "h":
            return HELP.format(parent=self.parent_frame, child=self.child_frame)
        return None

    def status_line(self):
        x, y, z = self.translation
        roll, pitch, yaw = self.rotation.as_euler("xyz", degrees=True)
        return (f"xyz [m]: {x:+.4f} {y:+.4f} {z:+.4f} | rpy [deg]: {roll:+7.2f} {pitch:+7.2f} {yaw:+7.2f} | "
                f"step: {self.translation_step * 1000:g} mm, {self.rotation_step:g} deg")

    def launch_block(self):
        """Final transform, formatted to paste over CAMERA_TF in launch/pc_with_camera_tf.launch.py."""
        x, y, z = self.translation
        qx, qy, qz, qw = self.rotation.as_quat()
        roll, pitch, yaw = self.rotation.as_euler("xyz", degrees=True)
        return (
            "\n# ===== Paste over CAMERA_TF in launch/pc_with_camera_tf.launch.py =====\n"
            f"# rpy [deg] (fixed-axis xyz): {roll:.4f}, {pitch:.4f}, {yaw:.4f}\n"
            "CAMERA_TF = {\n"
            f"    'x': {x:.6f}, 'y': {y:.6f}, 'z': {z:.6f},\n"
            f"    'qx': {qx:.8f}, 'qy': {qy:.8f}, 'qz': {qz:.8f}, 'qw': {qw:.8f},\n"
            f"    'frame_id': '{self.parent_frame}',\n"
            f"    'child_frame_id': '{self.child_frame}',\n"
            "}\n"
            "# To keep tuning from here:\n"
            f"#   ros2 run perception_tools_pkg interactive_tf_publisher --x {x:.6f} --y {y:.6f} --z {z:.6f} "
            f"--qx {qx:.8f} --qy {qy:.8f} --qz {qz:.8f} --qw {qw:.8f} "
            f"--parent-frame {self.parent_frame} --child-frame {self.child_frame}\n"
        )


# ===========================
# MAIN
# ===========================
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Hand-tune a static transform with the keyboard.")
    parser.add_argument("--parent-frame", default="PSM1_base_link")
    parser.add_argument("--child-frame", default="camera_color_optical_frame")
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--z", type=float, default=0.0)
    parser.add_argument("--qx", type=float, default=0.0)
    parser.add_argument("--qy", type=float, default=0.0)
    parser.add_argument("--qz", type=float, default=0.0)
    parser.add_argument("--qw", type=float, default=1.0)
    return parser.parse_args(argv)


def raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main(args=None):
    # Handle Ctrl-C / SIGTERM ourselves so the final transform is always printed
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)
    cli_args = parse_args(remove_ros_args(args if args is not None else sys.argv)[1:])

    if not sys.stdin.isatty():
        print("interactive_tf_publisher needs an interactive terminal (use ros2 run, not ros2 launch).")
        rclpy.try_shutdown()
        return

    node = InteractiveTfPublisher(
        parent_frame=cli_args.parent_frame,
        child_frame=cli_args.child_frame,
        translation=[cli_args.x, cli_args.y, cli_args.z],
        quaternion=[cli_args.qx, cli_args.qy, cli_args.qz, cli_args.qw],
    )
    print(HELP.format(parent=node.parent_frame, child=node.child_frame))
    print(node.status_line())

    # cbreak: read single keys without Enter, but keep Ctrl-C as SIGINT
    stdin_fd = sys.stdin.fileno()
    old_terminal_settings = termios.tcgetattr(stdin_fd)
    try:
        tty.setcbreak(stdin_fd)
        while True:
            readable, _, _ = select.select([stdin_fd], [], [], 0.1)
            if readable:
                # os.read (not sys.stdin.read) so held-down keys are not left in Python's input buffer
                output = node.handle_key(os.read(stdin_fd, 1).decode(errors="ignore"))
                if output is not None:
                    print(output)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_terminal_settings)
        print(node.launch_block())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
