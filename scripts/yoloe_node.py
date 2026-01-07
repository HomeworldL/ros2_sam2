#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yoloe_node.py

ROS2 node that continuously runs YOLOE object detection on incoming compressed images,
renders detection boxes and labels onto the image, and publishes the visualization as
a compressed image on /sam2/detection/compressed.

Behavior:
 - Subscribe to a CompressedImage input stream (configurable via ROS parameter).
 - Cache the latest frame in memory in image_callback; do not run detection inside the
   subscription callback to avoid blocking the DDS / executor.
 - Use a timer-driven processing loop to run YOLOE.detect on the latest cached frame
   at a configurable rate and publish the visualization as sensor_msgs/CompressedImage (JPEG).

Parameters:
 - image_topic (string): subscription topic for input compressed images
 - detection_topic (string): output topic for visualized detections (CompressedImage)
 - yoloe_model (string): YOLOE model filename or path
 - prompt_names (list[string]): textual prompts / class names for YOLOE text-conditioned detection
 - yoloe_conf (float): detection confidence threshold
 - process_rate (float): processing rate in Hz
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

import numpy as np
import cv2
import time
import os

try:
    from ultralytics import YOLOE
except Exception as e:
    raise Exception("Unable to import YOLOE: " + str(e))


class YOLOENode(Node):
    def __init__(self):
        super().__init__("yoloe_node")

        # ----------------------------
        # ROS parameters (configurable)
        # ----------------------------
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw/compressed")
        self.declare_parameter("detection_topic", "/sam2/detection/compressed")
        self.declare_parameter("yoloe_model", "yoloe-v8l-seg.pt")
        self.declare_parameter("prompt_names", ["person"])
        self.declare_parameter("yoloe_conf", 0.3)
        self.declare_parameter("process_rate", 10.0)  # Hz

        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.detection_topic = self.get_parameter("detection_topic").get_parameter_value().string_value
        self.yoloe_model_path = self.get_parameter("yoloe_model").get_parameter_value().string_value
        self.prompt_names = self.get_parameter("prompt_names").get_parameter_value().string_array_value
        self.yoloe_conf = self.get_parameter("yoloe_conf").get_parameter_value().double_value
        self.process_rate = self.get_parameter("process_rate").get_parameter_value().double_value

        self.get_logger().info(f"Subscribing image topic: {self.image_topic}")
        self.get_logger().info(f"Publishing detection visuals to: {self.detection_topic}")
        self.get_logger().info(f"YOLOE model: {self.yoloe_model_path}; prompts: {self.prompt_names}")
        self.get_logger().info(f"YOLOE confidence threshold: {self.yoloe_conf}; process_rate: {self.process_rate} Hz")

        # ----------------------------
        # CV bridge, subscribers, publishers
        # ----------------------------
        self.bridge = CvBridge()
        # subscribe to compressed image and cache latest frame (do not run heavy work in callback)
        self.sub_image = self.create_subscription(CompressedImage, self.image_topic, self.image_callback, 5)
        # publisher for detection visualization (CompressedImage)
        self.pub_vis = self.create_publisher(CompressedImage, self.detection_topic, 5)

        # storage for latest frame and header
        self.latest_bgr = None
        self.latest_header = None

        # ----------------------------
        # Initialize YOLOE model
        # ----------------------------
        self.get_logger().info("Loading YOLOE model...")
        self.yoloe = YOLOE(self.yoloe_model_path)

        # If the model supports text prompt configuration, set it
        if hasattr(self.yoloe, "get_text_pe") and hasattr(self.yoloe, "set_classes"):
            pe = self.yoloe.get_text_pe(self.prompt_names)
            self.yoloe.set_classes(self.prompt_names, pe)
            self.get_logger().info("YOLOE text prompts set.")

        # ----------------------------
        # Timer for processing loop
        # ----------------------------
        self.timer = self.create_timer(1.0 / float(self.process_rate), self.timer_callback)
        self.get_logger().info("YOLOE node initialized.")

    def image_callback(self, msg: CompressedImage):
        """
        Subscription callback: decode compressed image to BGR and cache the latest frame.
        This callback intentionally avoids running any heavy computation in order to
        minimize latency and avoid blocking the ROS executor.
        """
        # decode compressed -> BGR numpy array
        frame_bgr = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.latest_bgr = frame_bgr
        self.latest_header = msg.header

    def timer_callback(self):
        """
        Periodic processing callback: if a new frame exists, run YOLOE.predict on a
        copy of the frame, draw detection boxes and labels, and publish the visualization
        as a compressed JPEG image.
        """
        if self.latest_bgr is None:
            return

        # operate on a copy to avoid races with the subscriber callback
        frame_bgr = self.latest_bgr.copy()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Run YOLOE detection (returns a list-like results object)
        results = self.yoloe.predict(frame_rgb, conf=self.yoloe_conf)

        # If there are no detection results, still publish original image (optional)
        if len(results) == 0 or getattr(results[0], "boxes", None) is None:
            # publish original frame as compressed JPEG to keep downstream pipeline alive
            _, jpg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            out_msg = CompressedImage()
            if self.latest_header is not None:
                out_msg.header = self.latest_header
            out_msg.format = "jpeg"
            out_msg.data = jpg.tobytes()
            self.pub_vis.publish(out_msg)
            return

        # Parse boxes, scores and class indices if available
        r = results[0]
        boxes_np = None
        scores_np = None
        classes_np = None

        # Try to get xyxy boxes
        if hasattr(r, "boxes") and hasattr(r.boxes, "xyxy"):
            try:
                boxes_np = r.boxes.xyxy.cpu().numpy()
            except Exception:
                try:
                    boxes_np = np.array(r.boxes.xyxy)
                except Exception:
                    boxes_np = None

        # Try to get confidences and class indices
        if hasattr(r, "boxes") and hasattr(r.boxes, "conf"):
            try:
                scores_np = r.boxes.conf.cpu().numpy()
            except Exception:
                scores_np = None

        if hasattr(r, "boxes") and hasattr(r.boxes, "cls"):
            try:
                classes_np = r.boxes.cls.cpu().numpy().astype(int)
            except Exception:
                classes_np = None

        # Draw boxes and labels onto image (BGR)
        vis = frame_bgr
        n = 0 if boxes_np is None else boxes_np.shape[0]
        for i in range(n):
            x1, y1, x2, y2 = map(int, boxes_np[i][:4])
            score = float(scores_np[i]) if scores_np is not None else None
            cls_id = int(classes_np[i]) if classes_np is not None else None

            # choose color and label
            color = (0, 255, 0)  # default green BGR
            if cls_id is not None and cls_id < len(self.prompt_names):
                label_text = f"{self.prompt_names[cls_id]}"
            else:
                label_text = "obj"

            if score is not None:
                label_text = f"{label_text} {score:0.2f}"

            # draw box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness=2)

            # label background
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label_text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)

        # Compress to JPEG and publish
        success, jpg = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not success:
            self.get_logger().error("cv2.imencode failed for detection visualization.")
            return

        out_msg = CompressedImage()
        if self.latest_header is not None:
            out_msg.header = self.latest_header
        out_msg.format = "jpeg"
        out_msg.data = jpg.tobytes()
        self.pub_vis.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YOLOENode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
