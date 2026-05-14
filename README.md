# DRIVEWATCH — Real-Time Driver Drowsiness Detection

> A lightweight, webcam-based driver drowsiness detection system using
> facial landmark geometry and lightweight CNNs, running entirely on
> a consumer CPU at ~48 FPS with no specialised hardware.

---

## Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Models](#models)
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)
- [Configuration](#configuration)
- [Keyboard Controls](#keyboard-controls)
- [Known Limitations](#known-limitations)
- [Future Scope](#future-scope)

---

## Overview

DRIVEWATCH detects driver drowsiness in real time using only a standard
RGB webcam. It monitors three independent physiological signals:

| Signal | Method | Threshold |
|---|---|---|
| Eye closure duration | EAR geometry + TinyEyeNet CNN | ≥ 0.35s → ALERT, ≥ 2.0s → DROWSY |
| Yawn accumulation | MAR geometry + TinyMouthNet CNN | 3 yawns in 5 min → DROWSY |
| Sustained fatigue | PERCLOS (30s rolling window) | ≥ 35% → DROWSY |

The system outputs four progressive alert states:

```
NORMAL  →  ALERT  →  DROWSY  →  YAWN
```

All metrics are logged per-frame to CSV for post-session analysis.

---

## Demo

```
┌─────────────────────────────────────────────────────────────────────┐
│  DRIVEWATCH  v2.0          ⚡ DROWSY          12:34:56      48 FPS  │
├──────────────┬──────────────────────────────────┬───────────────────┤
│  TELEMETRY   │                                  │  ALERTS           │
│              │   ┌──────────────────────────┐   │                   │
│  ⌒EAR  ⌒MAR │   │                          │   │  ◯ Eye closure    │
│  arc gauges  │   │    LIVE WEBCAM FEED       │   │    ring timer     │
│              │   │    (with landmarks)       │   │                   │
│  ML-EYE ▬▬▬ │   │ ┌──┐              ┌──┐   │   │  ● ● ○  Yawns    │
│  ML-MTH ▬▬▬ │   │ │L │              │R │   │   │  2/3 in 5min     │
│  FUSED  ▬▬▬ │   │ │EY│              │EY│   │   │                   │
│  PERCLOS▬▬▬ │   │ └──┘              └──┘   │   │  PERCLOS bar      │
│              │   └──────────────────────────┘   │                   │
├──────────────┴──────────────────────────────────┴───────────────────┤
│  EAR 0.213   MAR 0.421   BLINK >0.35s  DROWSY >2.0s   [Q]uit       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## System Architecture

```
Webcam (640×480)
      │
      ▼
MediaPipe Face Mesh (468 landmarks)
      │
      ├──► EAR computation (6 eye landmarks per eye)
      ├──► MAR computation (4 mouth landmarks)
      ├──► Eye crops (16 ring landmarks per eye → 24×24 grayscale)
      └──► Mouth crop (20 landmarks → 64×64 grayscale)
                │
                ▼
     TinyEyeNet (ONNX)    TinyMouthNet (ONNX)
     0.051 ms/frame        0.149 ms/frame
                │
                ▼
     EAR-ML Fusion (0.70 × EAR + 0.30 × ML)
                │
                ▼
     ┌─────────────────────────────────┐
     │  Temporal Filters               │
     │  BlinkFilter   ≥ 0.35s gate    │
     │  YawnCounter   3 / 5 min       │
     │  PERCLOS       35% / 30s       │
     └─────────────────────────────────┘
                │
                ▼
     State Machine → NORMAL / ALERT / DROWSY / YAWN
                │
                ▼
     HUD Display + Audible Beep + CSV Log
```

---

## Project Structure

```
driver_drowsiness/
│
├── main_webcam_drowsiness.py      ← Entry point — run this
├── src/
│   ├── inference_engine.py        ← ONNX inference module
│   ├── train_eye_modal.py         ← TinyEyeNet training script
│   ├── train_yawn_modal.py        ← TinyMouthNet training script
│   ├── eye_data_prep.py           ← Eye dataset filter/balance/split
│   ├── collect_eye_data.py        ← Webcam eye data collector
│   └── export_eye_onnx.py         ← PyTorch → ONNX export
│
├── models/
│   └── onnx/
│       ├── eye_model.onnx         ← TinyEyeNet (23.2 KB)
│       └── mouth_model.onnx       ← TinyMouthNet (32.9 KB)
│
├── modals/
│   ├── eye/
│   │   └── best_eye_model.pth     ← Trained eye checkpoint
│   └── mouth/
│       └── best_mouth_model.pth   ← Trained mouth checkpoint
│
├── data/
│   ├── raw/
│   │   ├── eye/
│   │   │   ├── open/              ← Raw open eye images
│   │   │   └── closed/            ← Raw closed eye images
│   │   └── mouth/
│   │       ├── yawn/              ← Raw yawn mouth images
│   │       └── no_yawn/           ← Raw non-yawn mouth images
│   ├── filtered/eye/              ← After quality filter
│   ├── balanced/eye/              ← After 50/50 balancing
│   └── splits/
│       ├── eye/
│       │   ├── train/ val/ test/
│       └── mouth/
│           ├── train/ val/ test/
│
└── logs/
    └── drowsiness_session.csv     ← Per-frame session log
```

---

## Installation

### Requirements

| Requirement | Version |
|---|---|
| Python | 3.10 |
| OS (inference) | Windows 10/11 |
| OS (training) | WSL Ubuntu 22.04 (GPU) |
| GPU (training only) | NVIDIA CUDA-capable |

### Step 1 — Clone the repository

```bash
git clone https://github.com/yourname/driver-drowsiness.git
cd driver-drowsiness
```

### Step 2 — Create virtual environment (Windows)

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
# Windows inference environment
pip install opencv-python mediapipe numpy onnxruntime

# WSL training environment (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install scikit-learn matplotlib tqdm onnx onnxruntime onnxscript
```

### Step 4 — Verify ONNX models are present

```
models/onnx/eye_model.onnx
models/onnx/mouth_model.onnx
```

If not present, run the export script after training (see [Training](#training)).

---

## Usage

### Run the system

```bash
python main_webcam_drowsiness.py
```

A 980×560 window opens showing the live webcam feed with the full
telemetry HUD. The system begins detecting immediately.

### Keyboard Controls

| Key | Action |
|---|---|
| `Q` | Quit the application |
| `M` | Toggle MediaPipe face mesh overlay |
| `D` | Toggle debug panel (raw metric values) |
| `R` | Reset all counters (PERCLOS, yawn count, blink timer) |

---

## How It Works

### 1. Eye Aspect Ratio (EAR)

Computed from six MediaPipe landmarks per eye using the
Soukupová-Čech formula:

```
EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2 × ‖p1−p4‖)
```

Values below **0.21** indicate eye closure. Left and right EAR
are averaged into a single value.

### 2. Mouth Aspect Ratio (MAR)

Computed from four mouth landmarks:

```
MAR = ‖top−bottom‖ / ‖left−right‖
```

Values above **0.65** sustained for ≥ 1.5 seconds indicate a yawn.

### 3. EAR-ML Fusion

TinyEyeNet provides a machine learning confirmation signal.
The two are combined with EAR-dominant weighting:

```
Fused = 0.70 × EAR_score + 0.30 × ML_conf
```

The fused score crossing **0.45** marks `eye_closed_raw = True`.

### 4. Temporal Filtering

Three filters prevent false alerts:

```
BlinkFilter   — eye must be closed ≥ 0.35s continuously
YawnCounter   — 3 confirmed yawns within 5 minutes
PERCLOS       — ≥ 35% eye closed frames over 30 seconds
```

### 5. Alert State Machine

```python
DROWSY  →  blink_dur ≥ 2.0s  OR  PERCLOS ≥ 35%  OR  yawn_count ≥ 3
ALERT   →  blink_dur ≥ 0.35s  (but < 2.0s)
YAWN    →  single yawn just confirmed
NORMAL  →  none of the above
```

A **2.5-second linger window** prevents rapid state oscillation.

---

## Models

### TinyEyeNet

| Property | Value |
|---|---|
| Task | Binary eye state classification (open / closed) |
| Input | 1 × 24 × 24 grayscale |
| Architecture | 3 ConvBlocks (1→8→16→32) + GAP + Dropout(0.5) + Linear(32→2) |
| Parameters | **6,010** |
| ONNX size | 23.2 KB |
| Inference time | **0.051 ms/frame** (ONNX Runtime CPU) |
| Test accuracy | **98.61%** |

### TinyMouthNet

| Property | Value |
|---|---|
| Task | Binary yawn detection (yawn / no_yawn) |
| Input | 1 × 64 × 64 grayscale |
| Architecture | 4 ConvBlocks (1→16→32→64→128) + GAP + Linear(128→64→2) |
| Parameters | **105,778** |
| ONNX size | 32.9 KB |
| Inference time | **0.149 ms/frame** (ONNX Runtime CPU) |
| Test accuracy | **94.21%** |

---

## Dataset

### Eye Dataset

| Source | Description |
|---|---|
| CEW (Closed Eyes in the Wild) | Public webcam-style eye images |
| Self-collected | Webcam crops from 3 subjects via MediaPipe collector |

| Stage | Open | Closed | Total |
|---|---|---|---|
| Raw | 45,414 | 44,330 | 89,744 |
| After filter | 26,053 | 17,469 | 43,522 |
| After balance | 17,469 | 17,469 | 34,938 |
| Train (70%) | 12,228 | 12,228 | 24,456 |
| Val (15%) | 2,620 | 2,620 | 5,240 |
| Test (15%) | 2,621 | 2,621 | 5,242 |

### Mouth Dataset

Pre-cropped yawn/no-yawn images (2,934 train / 635 val / 639 test).

### Collecting Your Own Eye Data

```bash
python src/collect_eye_data.py
```

Sit at your laptop with adequate lighting. The script uses MediaPipe
to crop your eye regions in real time and saves them automatically.
Collect with eyes **open** first, then with eyes **closed**.

---

## Training

> Training requires WSL Ubuntu 22.04 with a CUDA-capable GPU.
> Inference on Windows requires only ONNX Runtime (no PyTorch).

### Step 1 — Prepare eye dataset

```bash
python src/eye_data_prep.py
```

This filters, balances, and splits the raw eye images into
`data/splits/eye/train`, `val`, and `test`.

### Step 2 — Train TinyEyeNet

```bash
python src/train_eye_modal.py
```

Trains for up to 80 epochs with early stopping (patience=15).
Best model saved to `modals/eye/best_eye_model.pth`.

### Step 3 — Train TinyMouthNet

```bash
python src/train_yawn_modal.py
```

Best model saved to `modals/mouth/best_mouth_model.pth`.

### Step 4 — Export to ONNX

```bash
# Export eye model only (keeps existing mouth model)
python src/export_eye_onnx.py
```

Exports to `models/onnx/eye_model.onnx` with constant folding.

---

## Results

### Model Performance

| Model | Test Accuracy | F1 | Parameters | Inference |
|---|---|---|---|---|
| TinyEyeNet (v1 MRL — failed) | 99.50%* | 0.9950 | 6,010 | 0.045ms |
| TinyEyeNet (v2 webcam) | **98.61%** | 0.9861 | 6,010 | 0.051ms |
| TinyMouthNet | **94.21%** | 0.9421 | 105,778 | 0.149ms |

*v1 achieved 99.50% on MRL test set but failed completely on webcam
(max confidence 0.5587) due to infrared domain gap.

### Live System

| Metric | Value |
|---|---|
| Average FPS | 47.8 |
| Pipeline latency | ~13.8ms (41.8% of 33ms budget) |
| ML confidence (eyes closed, webcam) | 0.9988–0.9995 |
| Fused score (eyes closed) | 0.9996–0.9998 |
| Blink filter false-alert rate | 0% (87/87 natural blinks filtered) |

### Key Finding

> A TinyEyeNet trained on **1,680 webcam-style images** completely
> outperformed a model trained on **24,456 infrared images** in real
> deployment. Domain match matters more than dataset size.

---

## Configuration

All parameters are in the `CFG` class at the top of
`main_webcam_drowsiness.py`:

```python
class CFG:
    # Model paths
    EYE_ONNX   = pathlib.Path("models/onnx/eye_model.onnx")
    MOUTH_ONNX = pathlib.Path("models/onnx/mouth_model.onnx")

    # Geometric thresholds
    EAR_THRESH = 0.21          # below = eye closed
    MAR_THRESH = 0.65          # above = mouth open

    # ML thresholds
    ML_EYE_THRESH   = 0.40    # minimum ML confidence to use
    ML_MOUTH_THRESH = 0.40

    # Fusion weights
    EAR_W = 0.70               # EAR-dominant
    ML_W  = 0.30
    FUSED_THRESH = 0.45        # fused score gate

    # Temporal thresholds
    BLINK_MIN_SEC  = 0.35      # min closure to raise ALERT
    DROWSY_MIN_SEC = 2.00      # min closure to raise DROWSY
    YAWN_MIN_SEC   = 1.50      # min mouth open to count as yawn
    YAWN_DROWSY_COUNT  = 3     # yawns needed for DROWSY
    YAWN_DROWSY_WINDOW = 300.0 # rolling window (seconds)

    # PERCLOS
    PERCLOS_WINDOW_SEC    = 30.0
    PERCLOS_DROWSY_THRESH = 0.35   # 35% closure = DROWSY

    # Alert
    ALERT_LINGER = 2.5         # seconds state persists after trigger
    BEEP_INTERVAL = 3.0        # seconds between audible beeps
```

---

## Known Limitations

| Limitation | Detail |
|---|---|
| Residual domain gap | ML eye confidence is 0.55–0.97 for open eyes (should be low). EAR-dominant fusion compensates but ML is not fully reliable. More training data across varied subjects would fix this. |
| Dark environments | System degrades below ~80 mean pixel brightness. Adequate indoor lighting required. |
| Glasses / sunglasses | Reflections interfere with MediaPipe landmark accuracy. |
| Single driver only | One face tracked per session. |
| Windows audio | Audible beep uses `winsound` — Linux/Mac sessions will not produce sound (detection still works). |

---

## Future Scope

- Collect 1,000+ images per class from 10+ persons to eliminate the residual domain gap in ML eye confidence
- Port ONNX models to Raspberry Pi 4 / NVIDIA Jetson Nano for embedded vehicle deployment
- Add head pose estimation (nodding detection) as a third drowsiness indicator
- Integrate per-driver EAR baseline calibration for improved threshold accuracy
- Multi-modal fusion with wearable heart rate variability data

---

## Tech Stack

```
OpenCV          — Webcam capture, image processing, HUD rendering
MediaPipe       — 468-point facial landmark detection
PyTorch         — Model training (GPU, WSL)
ONNX Runtime    — Real-time CPU inference (Windows)
NumPy           — Numerical computation throughout
scikit-learn    — Metrics, dataset splitting
Matplotlib      — Training curves, confusion matrix
```

---

## References

- Soukupová & Čech (2016) — Eye Aspect Ratio for blink detection
- Danisman et al. (2010) — PERCLOS as a fatigue metric
- Weng et al. (2017) — CNN eye classifier on MRL dataset
- Ghoddoosian et al. (2019) — Multi-signal drowsiness fusion
- Lugaresi et al. (2019) — MediaPipe Face Mesh
- Rasna et al. (2022) — ONNX Runtime benchmarking

---

## License

This project was developed as a Final Year B.Tech project.
For academic and research use only.