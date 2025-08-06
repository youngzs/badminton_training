# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered sports training analysis system that uses computer vision to analyze athlete movements and provide real-time feedback. The system has been completely rebuilt with a modern architecture.

## Current Architecture

### New System Structure (sports_analyzer/)
- **core/** - Core analysis modules
  - `motion_analyzer.py` - Real-time pose detection and analysis using MediaPipe
  - `feedback_system.py` - Intelligent feedback generation based on movement analysis
  - `video_processor.py` - Video stream processing and management
- **api/** - FastAPI backend service
  - `main.py` - REST API and WebSocket endpoints
- **frontend/** - Web interface
  - `index.html` - Single-page application for user interaction

### Legacy Code (for reference only)
- `badminton_pose.py` - Original pose detection implementation
- `spider.py` - Web scraper (not used in new system)

## Key Dependencies

- **FastAPI** - High-performance web framework
- **MediaPipe** - Google's pose detection library
- **OpenCV** - Video processing
- **NumPy/SciPy** - Numerical computations
- **WebSockets** - Real-time communication
- **Python 3.12.3** - Primary development language

## Development Commands

### Setup Environment
```bash
cd sports_analyzer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the System
```bash
# Quick start with script
chmod +x run.sh
./run.sh

# Or manually start API
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Access Points
- API Documentation: http://localhost:8000/docs
- Frontend: Open `sports_analyzer/frontend/index.html` in browser

## Architecture Details

### Motion Analysis Pipeline
1. Video capture from webcam/file/stream
2. Frame-by-frame pose detection (33 landmarks)
3. Joint angle calculation and optimization
4. Multi-dimensional scoring (angles, balance, fluidity, symmetry)
5. Real-time feedback generation
6. WebSocket streaming to frontend

### Scoring Algorithm Weights
- Joint angle accuracy: 40%
- Movement fluidity: 30%
- Body balance: 20%
- Timing accuracy: 10%

## Important Considerations

- Optimal camera distance: 2-3 meters from subject
- Good lighting required for accurate detection
- Full body should be visible in frame
- System supports multiple sport types with different optimal angle ranges
- Real-time processing requires decent CPU/GPU
- WebSocket connection for live stats updates

## API Key Endpoints

- `POST /api/v1/session/start` - Start training session
- `POST /api/v1/session/{id}/stop` - Stop and get feedback
- `GET /api/v1/session/{id}/stats` - Real-time statistics
- `POST /api/v1/analysis/upload` - Analyze video file
- `WS /ws/{session_id}` - WebSocket for live data

## Performance Tips

- Lower resolution for faster processing
- Use GPU-accelerated TensorFlow if available
- Limit frame history to prevent memory issues
- Adjust buffer sizes in VideoConfig for smoother streaming