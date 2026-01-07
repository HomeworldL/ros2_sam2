# ros2_sam2


A minimal ROS 2 interface for **SAM2 (Segment Anything 2)** real-time segmentation and tracking.

This repository provides a lightweight ROS 2 package (ecosystem-style) that can be built and used directly inside your workspace.  

It is based on the **segment-anything-2-real-time** library and supports real-time video segmentation (tracking-based segmentation).

## Tested Environment
- **Ubuntu 22.04**
- **ROS 2 Humble**
- **NVIDIA GPU (≥ 8 GB VRAM recommended)**
- **Intel RealSense Camera**

---

## Repository Overview

- `sam2_node`
  - Subscribes to compressed RGB images
  - Runs YOLOE once for first-frame detection
  - Initializes SAM2 with bounding-box prompts
  - Tracks objects frame-by-frame
  - Publishes merged binary masks as `sensor_msgs/CompressedImage`

- `mask_viewer`
  - Time-synchronized image + mask visualization
  - Overlays segmentation mask on RGB image
  - Publishes compressed visualization output

---

## Installation

> Replace `<your_conda>` and `<your_workspace>` with your own environment name and workspace path.

---

### 1. Create Workspace and Clone Repository
```bash
conda activate <your_conda>
cd <your_workspace>
mkdir -p src && cd src

git clone https://github.com/HomeworldL/ros2_sam2.git
cd ..
```
---

### 2. Install Python Dependencies


```bash
# PyTorch (CUDA 11.8 example)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

pip install ultralytics -U

# Install segment-anything-2-real-time
cd <other_dir>
git clone https://github.com/Gy920/segment-anything-2-real-time
cd segment-anything-2-real-time
pip install . --no-build-isolation

```
> `--no-build-isolation` is required to ensure correct interaction with the existing PyTorch / CUDA environment.

---

### 3. Download Pretrained Weights

```bash
cd src/ros2_sam2/checkpoints

./download_ckpts.sh
```

---

### 4. Configuration

Parameters are provided via ROS 2 YAML configuration files.

Important parameters:

- `image_topic`: input compressed image topic
- `mask_topic`: output mask topic
- `prompt_names`: list of object categories (YOLOE text prompts)
- `sam2_cfg`: **SAM2 configuration file**
- `sam2_checkpoint`: **SAM2 checkpoint path**
- `process_rate`: processing loop rate (Hz)


**Recommended real-time configuration:**

```yaml
sam2_cfg: "configs/sam2.1/sam2.1_hiera_t_512.yaml"
sam2_checkpoint: "/checkpoints/sam2.1_hiera_tiny.pt"
```

This configuration provides the best real-time performance.

---

### 5. Build
```bash
cd <your_workspace>
colcon build --symlink-install
```

---

## Running

### 1. Launch RealSense Camera
```bash
conda activate <your_conda>

ros2 launch realsense2_camera rs_launch.py \
    enable_sync:=true \
    align_depth.enable:=true \
    enable_color:=true \
    enable_depth:=true \
    pointcloud.enable:=false
```
### 2. Launch SAM2 and Mask Viewer

```bash
ros2 launch ros2_sam2 sam2.launch.py
```

> ⚠️ **First startup may be slow**
>
> Because the following options are enabled:
>
> - `compile_image_encoder: True`
> - `compile_memory_encoder: True`
> - `compile_memory_attention: True`
> - `vos_optimized = True`
>
> the first run includes model compilation and initialization. Please wait patiently.

---

### Performance

- Real-time segmentation at approximately **30 Hz**
- Actual performance depends on:
  - GPU model and VRAM
  - SAM2 configuration
  - Input image resolution
  - System load

---


## Notes

- `vos_optimized` can be enabled or disabled depending on accuracy vs latency requirements
- For strict real-time behavior, use compressed image topics and sensor-data QoS
- Tiny SAM2 checkpoints are recommended for online tracking

---


## TODO

- [ ] Add ROS 2 service interface for single-image segmentation
- [ ] Add point-cloud based segmentation support

## Acknowledgements

* [SAM2](https://github.com/facebookresearch/sam2)
* [segment-anything-2-real-time](https://github.com/Gy920/segment-anything-2-real-time)
