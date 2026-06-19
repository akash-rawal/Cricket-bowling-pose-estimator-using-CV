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

## Bowling Action Analysis - Logic & Data Structures

### 1. Keypoint Detection

The system extracts coordinates from the first detected person keypoint layout:

```python
results = model(frame, verbose=False)
kps = result.keypoints.data[0]  # shape: (N, 3) -> [x, y, conf]
```

Keypoint index mapping used:

* **0**: Shoulder (S) - Color: Blue
* **1**: Elbow (E) - Color: Orange
* **2**: Wrist (W) - Color: Green

Coordinates are stored in a dictionary `pts = dict[int, tuple[int, int]]` only if the detection confidence `conf > 0.3` and coordinates are valid.

### 2. Wrist Trail

```python
wrist_trail = deque(maxlen=300)  # circular buffer of (x, y) tuples
```

* **Jump Filter:** Rejects noisy or teleporting detections. A new wrist position is accepted only if the Euclidean distance between successive detections is less than `MAX_JUMP` (100 px).
* **Smoothing:** A centered moving average filter is applied over a 20-frame window to smooth the path.
* **Trail Rendering:** Renders a color gradient (orange to yellow) with line thickness fading from 2px to 6px as the trail ages.

### 3. Elbow Angle

```python
angle_history = deque(maxlen=200)  # stores float (degrees)
```

The elbow angle is calculated using vectors $\vec{u}$ (Shoulder to Elbow) and $\vec{v}$ (Wrist to Elbow):

* **Dot Product:** $d = \vec{u} \cdot \vec{v}$
* **Magnitudes:** $m_1 = \|\vec{u}\|$, $m_2 = \|\vec{v}\|$
* **Angle Formula:** $\theta = \arccos\left(\text{clamp}\left(\frac{d}{m_1 \cdot m_2}, -1, 1\right)\right) \times \frac{180}{\pi}$

*Fallback:* If any keypoint is missing, the last known elbow angle is repeated.

### 4. Wrist Speed

```python
speed_history = deque(maxlen=200)  # smoothed m/s values
speed_buffer  = deque(maxlen=7)    # rolling buffer for smoothing
```

* **Pixel-to-Meter Calibration:** Automatically calculated frame-by-frame based on the detected shoulder-to-wrist pixel distance compared to real-world arm length:
  $$\text{scale} = \frac{\text{ARM\_LENGTH\_M (2.5m)}}{\text{arm\_pixels}}$$
* **Speed Calculation:**
  $$\text{speed}_{\text{px}} = \sqrt{dx^2 + dy^2} \times \text{FPS}$$
  $$\text{speed}_{\text{ms}} = \text{speed}_{\text{px}} \times \text{scale}$$
  A 7-frame rolling average is applied to smooth out instantaneous spikes.

*Fallback:* If a frame contains a rejected wrist coordinate jump or keypoints are missing, the last known speed value is repeated.

### 5. Fan Lines

```python
sampled = smoothed_trail[::FAN_INTERVAL]   # every 25th point
```

To visualize the sweep, a line is drawn connecting the active `elbow_pt` to the historical wrist points sampled at every `FAN_INTERVAL` (25 frames) if the distance is less than 500 px. The colors fade from light-gray to white.

### 6. Analytics Panel

```python
panel = np.zeros((h, 420, 3), dtype=np.uint8)  # dark canvas, BGR
```

Two customized graphs are rendered side-by-side with the main video frame using `draw_graph()`:

* **Elbow Angle:** Sourced from `angle_history` (Y-max: 180°, Cyan color)
* **Wrist Speed:** Sourced from `speed_history` (Y-max: 20 m/s, Green color)

Each graph displays a translucent shaded area under the curve (20% opacity), a solid line chart, grid lines, and live value trackers in the top-right corner.

### 7. Combined Output Video

The telemetry panel is combined with the annotated video frame horizontally and saved:

```python
combined = np.hstack([frame, panel])   # shape: (h, w+420, 3)
out.write(combined)
```

At the end of processing, the system prints the final stats:
$$\text{Detection Rate} = \frac{\text{detected\_frames}}{\text{frame\_count}} \times 100\%$$

---
