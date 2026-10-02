import numpy as np
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


def numpy_to_pointcloud2(cloud_array, frame_id="map", stamp=None):
    """
    Convert a NumPy array to a ROS PointCloud2 message.

    Args:
        cloud_array (numpy.ndarray): The point cloud data as an Nx3 array.
        frame_id (str, optional): The frame ID for the PointCloud2 message. Defaults to "map".
        stamp (builtin_interfaces.msg.Time, optional): Time stamp to use. Defaults to zero.

    Returns:
        sensor_msgs.msg.PointCloud2: The PointCloud2 message.
    """
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    header = Header()
    header.frame_id = frame_id
    if stamp is not None:
        header.stamp = stamp

    points = np.asarray(cloud_array, dtype=np.float32)[:, :3]

    return point_cloud2.create_cloud(header, fields, points)
