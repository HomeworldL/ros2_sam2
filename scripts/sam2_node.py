#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sam2_node.py

Behavior:
 - Subscribe to a CompressedImage stream (input).
 - image_callback:
     Decode only the latest incoming image to BGR format and cache it in memory,
     together with its ROS header. No model inference is performed here.
 - Timer-based processing loop:
     * On the first frame, run YOLOE once to obtain bounding boxes and initialize
       the SAM2 predictor (load_first_frame + add_new_prompt).
     * On subsequent frames, call predictor.track() to obtain out_mask_logits.
 - Merge masks of all tracked objects into a single-channel binary image
   (foreground pixels = 255, background = 0), compress it as PNG, and publish
   as sensor_msgs/CompressedImage.

Parameters:
 - image_topic (string), default '/camera/camera/color/image_raw/compressed'
 - mask_topic (string), default '/sam2/mask/compressed'
 - yoloe_model (string)
 - prompt_names (list[string])
 - sam2_cfg (string)
 - sam2_checkpoint (string)
 - device (string), 'cuda' | 'cpu'
 - yoloe_conf (float)
 - process_rate (float), in Hz
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
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

try:
    from sam2.build_sam import build_sam2_camera_predictor
except Exception as e:
    raise Exception("Unable to import SAM2: " + str(e))

import torch
from ament_index_python.packages import get_package_share_directory

# Enable bfloat16 autocast globally (intended for CUDA execution)
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # Enable TF32 acceleration on Ampere GPUs
    # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class SAM2Node(Node):
    def __init__(self):
        super().__init__("sam2_node")

        # Declare ROS parameters (overridable via ros2 run / launch)
        self.declare_parameter(
            "image_topic", "/camera/camera/color/image_raw/compressed"
        )
        self.declare_parameter("mask_topic", "/sam2/mask/compressed")
        self.declare_parameter("yoloe_model", "yoloe-v8l-seg.pt")
        self.declare_parameter("prompt_names", ["person"])
        self.declare_parameter("sam2_cfg", "configs/sam2.1/sam2.1_hiera_t_512.yaml")
        self.declare_parameter("sam2_checkpoint", "/checkpoints/sam2.1_hiera_tiny.pt")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("yoloe_conf", 0.3)
        self.declare_parameter("process_rate", 30.0)  # Hz

        # Read parameters
        self.image_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value
        )
        self.mask_topic = (
            self.get_parameter("mask_topic").get_parameter_value().string_value
        )
        self.yoloe_model_path = (
            self.get_parameter("yoloe_model").get_parameter_value().string_value
        )
        self.prompt_names = (
            self.get_parameter("prompt_names").get_parameter_value().string_array_value
        )
        self.sam2_cfg = (
            self.get_parameter("sam2_cfg").get_parameter_value().string_value
        )
        self.sam2_ckpt_param = (
            self.get_parameter("sam2_checkpoint").get_parameter_value().string_value
        )
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.yoloe_conf = (
            self.get_parameter("yoloe_conf").get_parameter_value().double_value
        )
        self.process_rate = (
            self.get_parameter("process_rate").get_parameter_value().double_value
        )

        # If the checkpoint path is relative, resolve it w.r.t. the package share directory
        try:
            pkg_path = get_package_share_directory("ros2_sam2")
            self.sam2_ckpt = os.path.join(pkg_path, self.sam2_ckpt_param.lstrip("/"))
        except Exception:
            # Fall back to the raw parameter value if the package is not found
            self.sam2_ckpt = self.sam2_ckpt_param

        self.get_logger().info(f"image_topic: {self.image_topic}")
        self.get_logger().info(f"mask_topic: {self.mask_topic}")
        self.get_logger().info(
            f"yoloe_model: {self.yoloe_model_path}, prompt_names: {self.prompt_names}"
        )
        self.get_logger().info(
            f"SAM2 cfg: {self.sam2_cfg}, ckpt: {self.sam2_ckpt}, device: {self.device}"
        )
        self.get_logger().info(f"process_rate: {self.process_rate} Hz")

        # CV bridge, subscriber, and publisher
        self.bridge = CvBridge()

        # Subscribe to compressed color images
        self.sub_image = self.create_subscription(
            CompressedImage, self.image_topic, self.image_callback, 5
        )

        # Publish the merged mask as a compressed PNG image
        self.pub_mask = self.create_publisher(CompressedImage, self.mask_topic, 5)

        # Storage for the most recent frame and its header
        self.frame_bgr = None
        self.last_header = None

        # Models and state flags
        self.yoloe_model = None
        self.predictor = None
        self.initialized = False
        self.next_obj_id = 1

        # Initialize models
        self._init_models()

        # Timer-driven processing loop
        self.timer = self.create_timer(
            1.0 / float(self.process_rate), self.timer_callback
        )
        self.steps = 0

    def _init_models(self):
        self.get_logger().info("Loading YOLOE model...")
        self.yoloe_model = YOLOE(self.yoloe_model_path)

        # Configure text prompts if supported by the model
        if hasattr(self.yoloe_model, "get_text_pe") and hasattr(
            self.yoloe_model, "set_classes"
        ):
            pe = self.yoloe_model.get_text_pe(self.prompt_names)
            self.yoloe_model.set_classes(self.prompt_names, pe)
            self.get_logger().info("YOLOE text prompts configured.")

        self.get_logger().info("Building SAM2 predictor...")
        self.predictor = build_sam2_camera_predictor(
            self.sam2_cfg, self.sam2_ckpt, device=self.device, vos_optimized=True
        )
        self.get_logger().info("SAM2 predictor constructed successfully.")

    def image_callback(self, msg: CompressedImage):
        """
        Decode the incoming compressed image to BGR format and cache only
        the most recent frame. No inference or heavy computation is performed here.
        """
        cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.frame_bgr = cv_img
        self.last_header = msg.header

    def timer_callback(self):
        """
        Periodic processing of the cached frame:
         - First iteration:
             Run YOLOE once to obtain bounding boxes and initialize SAM2
             (load_first_frame + add_new_prompt).
         - Subsequent iterations:
             Call predictor.track(frame_rgb) to obtain out_mask_logits.
        Merge all object masks into a single-channel binary image, compress it
        as PNG, and publish it as a CompressedImage message.
        """
        ts1 = time.time()
        if self.frame_bgr is None:
            return  # No image available yet

        frame_bgr = self.frame_bgr
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Initialization stage: run YOLOE and add prompts to SAM2
        if not self.initialized:
            self.get_logger().info("First processing step: running YOLOE and initializing SAM2.")
            bboxes = []
            results = self.yoloe_model.predict(frame_rgb, conf=self.yoloe_conf)
            if len(results) > 0:
                r = results[0]
                # Try to extract bounding boxes from common YOLOE result formats
                if hasattr(r, "boxes") and hasattr(r.boxes, "xyxy"):
                    boxes_np = r.boxes.xyxy.cpu().numpy()
                    for bb in boxes_np:
                        x1, y1, x2, y2 = map(float, bb[:4])
                        bboxes.append([x1, y1, x2, y2])

                    self.get_logger().info(
                        f"YOLOE detected {len(bboxes)} boxes: {bboxes}"
                    )
                elif hasattr(r, "boxes") and hasattr(r.boxes, "data"):
                    try:
                        boxes_np = r.boxes.data
                        for bb in boxes_np:
                            x1, y1, x2, y2 = map(float, bb[:4])
                            bboxes.append([x1, y1, x2, y2])
                    except Exception:
                        pass

            # Abort if no bounding boxes could be obtained
            if len(bboxes) == 0:
                raise Exception(
                    "SAM initialization failed: YOLOE did not detect any objects."
                )

            # Load the first frame and register bounding-box prompts
            self.predictor.load_first_frame(frame_rgb)
            for bb in bboxes:
                x1, y1, x2, y2 = bb
                bbox_np = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                obj_id = self.next_obj_id
                self.next_obj_id += 1
                self.predictor.add_new_prompt(frame_idx=0, obj_id=obj_id, bbox=bbox_np)
                self.get_logger().info(
                    f"Added SAM2 prompt: obj_id={obj_id}, bbox={bbox_np.tolist()}"
                )

            self.initialized = True
            self.get_logger().info("SAM2 initialization complete. Switching to tracking mode.")
            return  # Do not publish a mask for the initialization frame

        # Tracking stage: call predictor.track() to obtain mask logits
        torch.cuda.synchronize()
        t0 = time.time()
        out_obj_ids, out_mask_logits = self.predictor.track(frame_rgb.copy())
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        self.get_logger().debug(
            f"track() elapsed {elapsed:.03f}s, objects={len(out_obj_ids)}"
        )

        # Merge multiple object masks into a single binary mask (uint8: 0 or 255)
        first_mask = out_mask_logits[0]
        if hasattr(first_mask, "cpu"):
            hw = first_mask.shape[-2:]
        else:
            hw = (
                first_mask.shape[-2:]
                if np.ndim(first_mask) >= 2
                else (frame_rgb.shape[0], frame_rgb.shape[1])
            )
        H, W = hw

        merged_mask = np.zeros((H, W), dtype=np.uint8)

        for mask_logits in out_mask_logits:
            if hasattr(mask_logits, "cpu"):
                mask_np = (mask_logits > 0.0).permute(1, 2, 0).cpu().numpy()
                if mask_np.ndim == 3:
                    mask2d = mask_np[:, :, 0]
                else:
                    mask2d = mask_np
            else:
                mask2d = (mask_logits > 0.0).astype(np.uint8)

            mask_bool = mask2d > 0
            merged_mask[mask_bool] = 255

        # Ensure the merged mask matches the original image resolution
        if (
            merged_mask.shape[0] != frame_rgb.shape[0]
            or merged_mask.shape[1] != frame_rgb.shape[1]
        ):
            merged_mask = cv2.resize(
                merged_mask,
                (frame_rgb.shape[1], frame_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Compress the mask as PNG and publish it as a CompressedImage
        success, png_data = cv2.imencode(".png", merged_mask)
        if not success:
            self.get_logger().error("cv2.imencode failed for mask PNG.")
            return

        msg = CompressedImage()
        if self.last_header is not None:
            msg.header = self.last_header
        msg.format = "png"
        msg.data = png_data.tobytes()
        self.pub_mask.publish(msg)

        self.steps += 1
        if self.steps % 100 == 0:
            self.get_logger().info(f"SAM2 tracked {self.steps} frames.")


def main(args=None):
    rclpy.init(args=args)
    node = SAM2Node()

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
