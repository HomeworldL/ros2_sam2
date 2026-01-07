#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mask_viewer.py

ROS 2 node for synchronized visualization of image and segmentation mask.

Subscriptions (configurable via parameters):
 - /camera/camera/color/image_raw/compressed   (sensor_msgs/CompressedImage)
 - /sam2/mask/compressed                       (sensor_msgs/CompressedImage)

This node uses message_filters.TimeSynchronizer to enforce strict timestamp
synchronization between the input image and mask streams.

The binary mask is overlaid on the RGB image and the blended result is:
 - displayed using OpenCV
 - published as a CompressedImage (PNG) on /sam2/viewer/compressed
"""

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage
import message_filters
from cv_bridge import CvBridge

import numpy as np
import cv2
import time
SHOW_IMAGE = False


class MaskViewer(Node):
    def __init__(self):
        super().__init__("mask_viewer")

        # ----------------------------
        # Parameters
        # ----------------------------
        self.declare_parameter(
            "image_topic", "/camera/camera/color/image_raw/compressed"
        )
        self.declare_parameter("mask_topic", "/sam2/mask/compressed")
        self.declare_parameter("queue_size", 10)

        self.image_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value
        )
        self.mask_topic = (
            self.get_parameter("mask_topic").get_parameter_value().string_value
        )
        self.queue_size = (
            self.get_parameter("queue_size").get_parameter_value().integer_value
        )

        self.get_logger().info(f"Image topic: {self.image_topic}")
        self.get_logger().info(f"Mask topic:  {self.mask_topic}")
        self.get_logger().info(f"TimeSynchronizer queue size: {self.queue_size}")

        # ----------------------------
        # ROS interfaces
        # ----------------------------
        self.bridge = CvBridge()

        self.sub_image = message_filters.Subscriber(
            self, CompressedImage, self.image_topic
        )
        self.sub_mask = message_filters.Subscriber(
            self, CompressedImage, self.mask_topic
        )

        # Strict timestamp synchronization
        self.ts = message_filters.TimeSynchronizer(
            [self.sub_image, self.sub_mask], self.queue_size
        )
        self.ts.registerCallback(self.synced_callback)

        # Publisher for blended visualization
        self.pub_viewer = self.create_publisher(
            CompressedImage, "/sam2/viewer/compressed", 5
        )

        # ----------------------------
        # Visualization
        # ----------------------------
        if SHOW_IMAGE:
            cv2.namedWindow("Image+Mask", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Image+Mask", 640, 480)

    def synced_callback(
        self,
        img_msg: CompressedImage,
        mask_msg: CompressedImage,
    ):
        """
        Callback executed when an image and mask with identical timestamps
        are received.

        Steps:
         1. Decode compressed RGB image.
         2. Decode compressed mask and convert to binary (0 / 255).
         3. Overlay mask on the image using a fixed color.
         4. Display and publish the blended visualization.
        """

        # Decode RGB image (BGR format for OpenCV)
        img_bgr = self.bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

        # Decode mask as single-channel image
        mask_gray = self.bridge.compressed_imgmsg_to_cv2(mask_msg)

        # Ensure mask is binary uint8 {0, 255}
        if mask_gray.dtype != np.uint8:
            mask_gray = (255.0 * mask_gray.astype(np.float32)).astype(np.uint8)

        _, mask_bin = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)

        # Resize mask if spatial resolution differs
        if mask_bin.shape[:2] != img_bgr.shape[:2]:
            mask_bin = cv2.resize(
                mask_bin,
                (img_bgr.shape[1], img_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Create colored overlay (green mask)
        overlay = np.zeros_like(img_bgr, dtype=np.uint8)
        pink = (255, 0, 255)  # BGR
        overlay[mask_bin > 0] = pink

        alpha = 0.3
        blended = cv2.addWeighted(img_bgr, 1.0, overlay, alpha, 0)

        # Display visualization
        if SHOW_IMAGE:
            cv2.imshow("Image+Mask", blended)
            cv2.waitKey(1)

        # Publish compressed visualization
        success, png_data = cv2.imencode(".png", blended)
        if success:
            out_msg = CompressedImage()
            out_msg.header = img_msg.header
            out_msg.format = "png"
            out_msg.data = png_data.tobytes()
            self.pub_viewer.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MaskViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Do not access the logger after destroy_node()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
