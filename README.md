# YOLO_3D Project

This project implements a 3D human pose estimation system using YOLO and deep learning techniques.

## Project Structure

```
.
├── common/               # Core functionality and utilities
│   ├── arguments.py     # Command line argument handling
│   ├── camera.py        # Camera parameters and operations
│   ├── generators.py    # Data generation utilities
│   ├── h36m_dataset.py  # Human3.6M dataset handler
│   ├── model.py         # Neural network model definitions
│   ├── utils.py         # General utility functions
│   └── visualization.py # Visualization tools
├── tool/                # Implementation tools
│   ├── utils.py         # Utility functions
│   ├── video.py        # Video processing tools
│   ├── yolo_realtime.py # Real-time YOLO detection
│   └── yolo_realtime_test.py # Testing script
└── yolo_detectors/     # YOLO detection implementations
    ├── yolov8_detection.py # YOLOv8 detector
    └── yolov8_npz.py      # NPZ file handling for YOLOv8
```

## Features

- Real-time 3D human pose estimation
- YOLOv8 integration for object detection
- Support for video processing
- Human3.6M dataset compatibility
- Camera calibration and handling

## Requirements

- Python
- YOLOv8
- OpenCV
- PyTorch (recommended)

## Quick Start

1. Install the required dependencies
2. Configure the camera parameters if needed
3. Run real-time detection:
   ```bash
   python tool/yolo_realtime.py
   ```

## Directory Description

- `common/`: Contains core functionality including dataset handling, model definitions, and utility functions
- `tool/`: Implementation tools for real-time processing and utilities
- `yolo_detectors/`: YOLO-specific detection implementations
- `weights/`: Directory for storing model weights (not included in repository)

## Notes

- Make sure to download and place the required model weights in the `weights` directory
- Check camera calibration settings before running real-time detection
- Refer to individual Python files for detailed documentation