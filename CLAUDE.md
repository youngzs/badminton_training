# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Badminton video pose detection and analysis tool using YOLOv8 and MediaPipe. The project provides both CLI and GUI interfaces to analyze badminton videos by detecting human pose skeletons, tracking sports equipment (shuttlecock, racket), and generating annotated output videos.

## Core Architecture

### Three Main Entry Points

1. **`pose_detection.py`** - Basic CLI implementation
   - Simple YOLOv8-pose detection
   - 17 COCO keypoints with skeleton rendering
   - Video processing with optional keyframe extraction

2. **`pose_detection_advanced.py`** - Enhanced CLI with multi-model support
   - Supports both YOLO (n/s/m/l/x variants) and MediaPipe
   - Object detection for ball/racket using YOLOv8
   - Performance metrics and FPS tracking
   - Audio merging using ffmpeg/moviepy

3. **`pose_detection_gui.py`** - Tkinter GUI wrapper
   - Inherits from `BadmintonAnalyzer` in `pose_detection_advanced.py`
   - Multi-threaded processing with progress callbacks
   - Configuration persistence via `gui_config.json`

### Class Hierarchy

```
PoseDetectorBase (abstract)
├── YOLOPoseDetector - YOLO-based pose detection
└── MediaPipePoseDetector - MediaPipe-based pose detection

ObjectDetector - YOLOv8 object detection for balls/rackets

BadmintonAnalyzer - Main video processor
└── CustomBadmintonAnalyzer - GUI-enhanced with callbacks
```

### Key Design Patterns

- **Strategy Pattern**: Pose detector selection (YOLO vs MediaPipe)
- **Template Method**: Common skeleton drawing with customizable parameters
- **Observer Pattern**: Progress/status callbacks for GUI integration

## Development Environment

### Setting Up

```bash
# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### GPU/CUDA Setup (Critical for Training Version)

**Common Issue:** PyTorch installs CPU version by default, GPU shows as unavailable.

**Quick Fix:**
```bash
# Automated fix
fix_gpu.bat

# Or manual install
install_cuda_pytorch.bat
```

**Verification:**
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
# Expected: CUDA: True
```

**Key Files:**
- `CUDA_INSTALL_GUIDE.md` - Detailed CUDA installation guide
- `QUICKSTART.md` - 5-minute setup guide
- `fix_gpu.bat` - Automatic GPU troubleshooting
- `install_cuda_pytorch.bat` - CUDA PyTorch installer
- `check_nvidia.bat` - Full environment diagnostics

### Virtual Environment

The project uses a Python virtual environment in `venv/`. All development should be done with this activated to ensure dependency isolation.

**Important:** Ensure PyTorch is installed with CUDA support for GPU acceleration. The training version (`pose_detection_training.py`) requires CUDA for optimal performance (10x speedup).

## Common Commands

### Running the Application

```bash
# Basic CLI usage
python pose_detection.py video.mp4 -o output.mp4

# Advanced CLI with MediaPipe
python pose_detection_advanced.py video.mp4 --pose-model mediapipe

# Advanced CLI with custom ball detection confidence
python pose_detection_advanced.py video.mp4 --ball-confidence 0.1 --object-model-size m

# Model comparison
python pose_detection_advanced.py video.mp4 --compare yolo-n,yolo-m,mediapipe

# Launch GUI
python pose_detection_gui.py

# Training mode with GPU acceleration (NEW)
python pose_detection_training.py video.mp4

# 2x slow motion for technique analysis
python pose_detection_training.py video.mp4 --slowmo 0.5 -o output_slow.mp4

# 4x slow motion with pose analysis
python pose_detection_training.py video.mp4 --slowmo 0.25 --analyze

# Benchmark GPU performance
python benchmark_gpu.py video.mp4 --frames 100
```

### Building Executable

```bash
# Using the build script
python build_exe.py

# Or manually with PyInstaller
pyinstaller --name=BadmintonVideoAnalyzer --onefile --windowed --noconfirm --clean pose_detection_gui.py
```

## Architecture Details

### Video Processing Pipeline

1. **Frame Capture**: OpenCV VideoCapture reads input video
2. **Pose Detection**: YOLO or MediaPipe processes each frame
3. **Object Detection**: YOLOv8 detects balls/rackets (optional)
4. **Skeleton Rendering**: Custom drawing with configurable line width/keypoint size
5. **Frame Writing**: OpenCV VideoWriter saves annotated frames
6. **Audio Merging**: ffmpeg or moviepy merges original audio to output

### Model Management

- **YOLO Models**: Auto-downloaded to `%USERPROFILE%\.ultralytics\weights\` on Windows
- **Model Variants**: n (nano, ~6MB) → s (small) → m (medium) → l (large) → x (extra-large, ~68MB)
- **Trade-offs**: Larger models = higher accuracy but slower processing

### Critical Detection Parameters

- **General confidence**: 0.25 (default for most objects)
- **Ball confidence**: 0.15 (lower threshold since shuttlecocks move fast and are small)
- **Person min confidence**: 0.7 (filters background people in GUI mode)

### Audio Handling Strategy

The code implements a two-stage audio merge:
1. Process video → temp file without audio
2. Merge original audio → final output
3. Attempts moviepy first, falls back to ffmpeg subprocess

### GUI-Specific Implementation

- **Threading**: Video processing runs in daemon thread to prevent UI blocking
- **Callbacks**: `progress_callback` and `status_callback` for real-time updates
- **Stop Signal**: Lambda function `stop_flag` for graceful interruption
- **Config Persistence**: JSON serialization of all settings

## Important Considerations

### Windows Console Encoding

The code includes extensive UTF-8 setup for Windows console output (lines 11-47 in each file). This handles Chinese character display in terminal output. The GUI version (pose_detection_gui.py) includes null-checks since stdout/stderr may be None in windowed mode.

### Model Download on First Run

First execution triggers automatic model downloads. Users need internet connectivity. Models are cached globally, so subsequent runs are faster.

### ffmpeg Dependency

Audio merging requires either:
- ffmpeg installed globally (preferred)
- moviepy with imageio-ffmpeg package

The build configuration includes `--collect-all=imageio_ffmpeg` to bundle ffmpeg in the executable.

### COCO Keypoint Format

YOLO uses 17 COCO keypoints:
- Head: nose, eyes, ears (0-4)
- Arms: shoulders, elbows, wrists (5-10)
- Torso: hips (11-12)
- Legs: knees, ankles (13-16)

MediaPipe provides 33 keypoints but gets mapped to COCO 17 for consistency.

### Object Detection Specifics

Ball detection uses lower confidence threshold (0.15 vs 0.25) because:
- Shuttlecocks are small
- Fast motion causes blur
- Lower threshold = more detections but more false positives

The code filters duplicates by tracking `seen_boxes` to avoid redundant detections.

## Testing Approach

No formal test suite exists. Manual testing workflow:
1. Test with sample badminton video
2. Verify skeleton drawing quality
3. Check ball/racket detection accuracy
4. Confirm audio preservation in output
5. Test GUI controls and progress updates

## Performance Notes

- YOLOv8n-pose: 30-50 FPS (recommended for real-time)
- YOLOv8m-pose: 15-25 FPS (good accuracy/speed balance)
- MediaPipe: 10-20 FPS (best for single-person detailed analysis)

Process multiple videos in batch by scripting calls to the CLI version.

## Training-Specific Features (pose_detection_training.py)

### GPU Optimization Strategy

The training version implements aggressive GPU optimization for RTX 4090:

1. **FP16 Half-Precision**: 2x speed improvement with minimal accuracy loss
2. **Batch Processing**: Process multiple frames simultaneously (configurable batch_size)
3. **CUDA Streams**: Async GPU operations for better throughput
4. **Memory Pinning**: Faster CPU-GPU transfers

### Performance Expectations (RTX 4090)

- **YOLOv8n-pose + FP16**: 100-150 FPS (real-time capable)
- **YOLOv8m-pose + FP16**: 50-80 FPS (good for training analysis)
- **YOLOv8l-pose + FP16**: 30-50 FPS (high accuracy)

CPU fallback: ~10-20 FPS (significantly slower, not recommended for training)

### Slow Motion Implementation

Uses linear frame interpolation (blending) for generating intermediate frames:
- `--slowmo 0.5` = 2x slower (1 interpolated frame between each original)
- `--slowmo 0.25` = 4x slower (3 interpolated frames between each original)

Production consideration: Replace with RIFE or FILM deep learning models for better quality.

### Pose Analysis Features

The `PoseAnalyzer` class provides:

1. **Angle Calculation**: Elbow, knee, torso angles for technique verification
2. **Issue Detection**: Automated flagging of common form problems
3. **Correction Suggestions**: Actionable feedback for students
4. **Reference Comparison**: Compare student pose against coach demonstrations (future)

### Key Metrics Tracked

- Right/left elbow angle (arm extension during swing)
- Knee flexion (footwork and balance)
- Torso lean angle (posture during shots)
- Shoulder rotation (power generation)

### Training Workflow

1. Record student performing technique (normal speed)
2. Process with GPU-accelerated detection
3. Generate slow-motion version with pose overlay
4. AI analyzes angles and flags issues
5. Save keyframes at critical moments for review
6. Coach reviews annotated video with student

## Code Style

- Chinese comments for user-facing strings
- English for technical documentation
- Type hints used in function signatures
- docstrings follow Google style (brief description → Args → Returns)
