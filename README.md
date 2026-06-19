<!-- markdownlint-disable MD033 -->

# Cricket Bowler Biomechanical Analyzer Using YOLOv8-Pose

An advanced computer-vision-based biomechanics analysis system designed to track, map, and evaluate cricket bowling actions from video footage. Utilizing a custom fine-tuned **YOLOv8x-Pose** architecture, this tool extracts high-precision joint coordinates to perform real-time kinematic calculations, generating a side-by-side biomechanical overlay and telemetry graphs.

---

## Demo & Inference Videos

Below are the video representations of the system in action. You can see the progression from raw footage to isolated biomechanical mapping and the final telemetry dashboard.

| Raw Input Video | Biomechanical Analysis |
| :---: | :---: |
| <video src="https://storage.googleapis.com/labellerr-cdn/%20%201%20cricket/clip.webm" controls width="100%"></video><br><a href="https://storage.googleapis.com/labellerr-cdn/%20%201%20cricket/clip.webm" target="_blank">Download / Watch Raw</a> | <video src="https://storage.googleapis.com/labellerr-cdn/%20%201%20cricket/output_analysis2.webm" controls width="100%"></video><br><a href="https://storage.googleapis.com/labellerr-cdn/%20%201%20cricket/output_analysis2.webm" target="_blank">Download / Watch Analysis</a> |

---

## Key Features

1. **Precision 3-Point Keypoint Tracking**
   * High-accuracy detection of key skeletal joints: **Shoulder (S)**, **Elbow (E)**, and **Wrist (W)**.
   * Confidence threshold filtering (`CONF_THRESHOLD = 0.3`) to prevent ghost/false detections.

2. **Wrist Motion Path & Sweep Schematic**
   * Real-time wrist trajectory mapping over a 300-frame history window.
   * **Jump Filter:** Detects and discards noisy coordinate jumps exceeding 100px.
   * **Smoothing Engine:** Applies a 20-frame centered moving average to ensure clean visual motion trails.
   * **Sweep Fan Lines:** Connects the elbow to periodic wrist points to illustrate the arm-swing arc.

3. **Dynamic kinematic Analytics**
   * **Elbow Joint Angle:** Calculates the live elbow flex/extension angle in degrees using vector dot products.
   * **Wrist Speed (m/s):** Automatically converts pixel velocities to meters per second by using the user's arm length as a spatial calibration reference (standard `ARM_LENGTH_M = 2.5` meters).
   * **Speed Smoothing:** Processes velocity with a 7-frame rolling average filter to prevent instantaneous acceleration spikes.

4. **Telemetry HUD Overlay**
   * Side-by-side dashboard featuring:
     * **Elbow Angle graph** (0° - 180° range) with a live readout.
     * **Wrist Speed graph** (0 - 20 m/s range) with a live readout.
     * Translucent area curves, gridlines, and frame trackers.

---

## Project Structure

```bash
Cricket/
├── annotations/            # Dataset COCO format annotation files
├── dataset/                # Train/Val split directory (generated during prep)
│   ├── train/              # 85 images & labels
│   └── val/                # 11 images & labels
├── bowling_env/            # Python virtual environment (ignored by git)
├── runs/                   # YOLOv8 training outputs, plots, and weights (ignored by git)
├── weights/                # Best fine-tuned model weights (best.pt)
├── Cricket_bowler_Analyzer_using_yolov8_pose.ipynb  # Primary code notebook
├── bowling_pose.yaml       # YOLO dataset configuration
├── clip.mp4                # Raw input video clip
├── output_analysis2.mp4    # Generated output video with HUD dashboard
└── README.md               # Project documentation
```

---

## Installation & Setup

### 1. Clone & Set Up Environment

Ensure you have Python 3.10+ installed.

```bash
# Clone the repository
git clone https://github.com/akash-rawal/Cricket-bowler-pose-estimator-using-CV.git
cd Cricket-bowler-pose-estimator-using-CV

# Create virtual environment
python -m venv bowling_env

# Activate environment
# On Windows:
bowling_env\Scripts\activate
# On Linux/macOS:
source bowling_env/bin/activate

# Install required dependencies
pip install ultralytics opencv-python numpy tqdm ipykernel
```

### 2. Dataset Preparation

The system relies on a 3-point keypoint layout. You can convert COCO annotations to YOLO format using the notebook:

* Input: `annotations/coco_annotations.json`
* Output split: `dataset/train` and `dataset/val`

### 3. Model Training

The dataset configuration is defined in `bowling_pose.yaml`:

```yaml
path: ../dataset
train: train/images
val: val/images

names:
  0: bowler

# Keypoint shape: [number of keypoints, dimensions (x, y, visibility)]
kpt_shape: [3, 3]
```

To run training:

Open `Cricket_bowler_Analyzer_using_yolov8_pose.ipynb` and execute the training cell (100 epochs, AdamW optimizer, batch size 4).

---

## Running Inference

To run the biomechanical analyzer on any video file (e.g. `clip.mp4`):

```python
from ultralytics import YOLO
import cv2

# Load the fine-tuned model
model = YOLO("weights/best.pt")

# Run predict
results = model.predict(source="clip.mp4", show=True, save=True)
```

To run the analyzer with the **side-by-side Telemetry HUD** overlay, run the main analysis cell in the notebook. It will read `clip.mp4` and generate `output_analysis2.mp4` with real-time graphs.

---

## Biomechanical Math

* **Angle Calculation:**
  $$\theta = \arccos\left(\frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}\right)$$
  where $\vec{u}$ is the shoulder-to-elbow vector, and $\vec{v}$ is the wrist-to-elbow vector.

* **Pixel-to-Meter Calibration:**
  $$\text{Scale (px to m)} = \frac{\text{Real Arm Length (m)}}{\text{Detected Arm Length (px)}}$$

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
