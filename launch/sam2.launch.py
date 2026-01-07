#!/usr/bin/env python3
"""
Launch both sam2_node and mask_viewer with parameter files.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('ros2_sam2')

    sam2_params = os.path.join(pkg_share, 'config', 'sam2_node.yaml')
    viewer_params = os.path.join(pkg_share, 'config', 'mask_viewer.yaml')

    # Node for SAM2 pipeline (yoloe + sam2)
    sam2_node = Node(
        package='ros2_sam2',
        executable='sam2_node.py',
        name='sam2_node',
        output='screen',
        parameters=[sam2_params]
    )

    # Node for synchronized viewer
    viewer_node = Node(
        package='ros2_sam2',
        executable='mask_viewer.py',
        name='mask_viewer',
        output='screen',
        parameters=[viewer_params]
    )

    return LaunchDescription([
        sam2_node,
        viewer_node,
    ])
