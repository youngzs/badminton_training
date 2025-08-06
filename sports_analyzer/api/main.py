"""
FastAPI主应用
提供REST API和WebSocket接口
"""

from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import json
import uuid
from datetime import datetime
import cv2
import numpy as np
import base64
import io
from pathlib import Path

# 导入核心模块
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.motion_analyzer import MotionAnalyzer, SportType
from core.feedback_system import FeedbackSystem
from core.video_processor import VideoProcessor, VideoConfig, VideoSource, VideoAnalyzer


# 创建FastAPI应用
app = FastAPI(
    title="Sports Analyzer API",
    description="AI-powered sports training analysis system",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境需要限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 全局变量存储活动会话
active_sessions: Dict[str, VideoAnalyzer] = {}


# Pydantic模型
class SessionConfig(BaseModel):
    sport_type: str = "badminton"
    video_source: str = "webcam"
    enable_recording: bool = False
    user_id: Optional[str] = None


class FrameAnalysis(BaseModel):
    timestamp: float
    score: float
    issues: List[str]
    suggestions: Optional[List[str]] = None


class TrainingSession(BaseModel):
    session_id: str
    user_id: Optional[str]
    sport_type: str
    start_time: datetime
    duration: float
    avg_score: float
    total_frames: int


class FeedbackResponse(BaseModel):
    overall_score: float
    level: str
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[Dict[str, str]]
    progress_notes: Optional[str]


# API端点
@app.get("/")
async def root():
    """API根路径"""
    return {
        "message": "Sports Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "session": "/api/v1/session",
            "analysis": "/api/v1/analysis"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/v1/session/start")
async def start_session(config: SessionConfig):
    """开始新的训练会话"""
    session_id = str(uuid.uuid4())
    
    try:
        # 配置视频源
        if config.video_source == "webcam":
            video_config = VideoConfig(
                source_type=VideoSource.WEBCAM,
                enable_recording=config.enable_recording,
                recording_path=f"recordings/{session_id}.mp4" if config.enable_recording else None
            )
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Unsupported video source"}
            )
        
        # 创建分析器
        analyzer = VideoAnalyzer()
        
        # 设置分析器
        if not analyzer.setup(video_config):
            raise HTTPException(status_code=500, detail="Failed to initialize video source")
        
        # 保存会话
        active_sessions[session_id] = analyzer
        
        # 启动分析（在后台运行）
        asyncio.create_task(analyzer.start_analysis())
        
        return {
            "session_id": session_id,
            "status": "started",
            "config": config.dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/session/{session_id}/stop")
async def stop_session(session_id: str):
    """停止训练会话"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    analyzer = active_sessions[session_id]
    
    # 停止分析并获取反馈
    feedback = analyzer.stop_analysis()
    
    # 清理会话
    del active_sessions[session_id]
    
    if feedback:
        return FeedbackResponse(
            overall_score=feedback.overall_score,
            level=feedback.level.value,
            strengths=feedback.strengths,
            weaknesses=feedback.weaknesses,
            suggestions=[
                {
                    "title": s.title,
                    "description": s.description,
                    "drill": s.drill_recommendation
                }
                for s in feedback.suggestions
            ],
            progress_notes=feedback.progress_notes
        )
    
    return {"message": "Session stopped", "session_id": session_id}


@app.get("/api/v1/session/{session_id}/stats")
async def get_session_stats(session_id: str):
    """获取会话统计信息"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    analyzer = active_sessions[session_id]
    stats = analyzer.get_current_stats()
    
    return stats


@app.post("/api/v1/analysis/upload")
async def analyze_video(file: UploadFile = File(...)):
    """分析上传的视频文件"""
    # 检查文件类型
    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    # 保存上传的文件
    upload_id = str(uuid.uuid4())
    upload_path = f"uploads/{upload_id}_{file.filename}"
    
    Path("uploads").mkdir(exist_ok=True)
    
    with open(upload_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    try:
        # 配置视频分析
        video_config = VideoConfig(
            source_type=VideoSource.FILE,
            source_path=upload_path
        )
        
        # 创建分析器
        analyzer = VideoAnalyzer()
        
        if not analyzer.setup(video_config):
            raise HTTPException(status_code=500, detail="Failed to process video")
        
        # 运行分析
        await analyzer.start_analysis()
        
        # 获取反馈
        feedback = analyzer.stop_analysis()
        
        if feedback:
            return FeedbackResponse(
                overall_score=feedback.overall_score,
                level=feedback.level.value,
                strengths=feedback.strengths,
                weaknesses=feedback.weaknesses,
                suggestions=[
                    {
                        "title": s.title,
                        "description": s.description,
                        "drill": s.drill_recommendation
                    }
                    for s in feedback.suggestions
                ],
                progress_notes=feedback.progress_notes
            )
        
        return {"message": "Analysis complete", "upload_id": upload_id}
        
    finally:
        # 清理上传的文件
        Path(upload_path).unlink(missing_ok=True)


@app.post("/api/v1/analysis/frame")
async def analyze_frame(image: UploadFile = File(...)):
    """分析单帧图像"""
    # 读取图像
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    
    # 创建分析器
    analyzer = MotionAnalyzer(SportType.BADMINTON)
    
    # 分析帧
    motion_frame = analyzer.process_frame(frame, 0.0)
    
    if motion_frame:
        return FrameAnalysis(
            timestamp=motion_frame.timestamp,
            score=motion_frame.score,
            issues=motion_frame.issues
        )
    
    raise HTTPException(status_code=500, detail="Failed to analyze frame")


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket端点for实时通信"""
    await websocket.accept()
    
    try:
        # 检查会话
        if session_id not in active_sessions:
            await websocket.send_json({"error": "Session not found"})
            await websocket.close()
            return
        
        analyzer = active_sessions[session_id]
        
        # 发送实时数据
        while session_id in active_sessions:
            # 获取当前统计
            stats = analyzer.get_current_stats()
            
            # 发送数据
            await websocket.send_json({
                "type": "stats",
                "data": stats
            })
            
            # 等待一段时间
            await asyncio.sleep(0.5)
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


@app.get("/api/v1/sports")
async def get_supported_sports():
    """获取支持的运动类型"""
    return {
        "sports": [
            {"id": "badminton", "name": "羽毛球", "description": "羽毛球动作分析"},
            {"id": "tennis", "name": "网球", "description": "网球动作分析"},
            {"id": "basketball", "name": "篮球", "description": "篮球投篮分析"},
            {"id": "golf", "name": "高尔夫", "description": "高尔夫挥杆分析"},
            {"id": "yoga", "name": "瑜伽", "description": "瑜伽姿势分析"},
            {"id": "running", "name": "跑步", "description": "跑步姿态分析"}
        ]
    }


@app.get("/api/v1/user/{user_id}/sessions")
async def get_user_sessions(user_id: str, limit: int = 10):
    """获取用户训练记录"""
    # 这里需要连接数据库获取真实数据
    # 示例返回
    return {
        "user_id": user_id,
        "sessions": [
            {
                "session_id": "example-123",
                "date": "2024-01-20",
                "sport_type": "badminton",
                "duration": 1800,
                "avg_score": 85.5
            }
        ]
    }


@app.get("/api/v1/user/{user_id}/progress")
async def get_user_progress(user_id: str, days: int = 30):
    """获取用户进度数据"""
    # 这里需要连接数据库获取真实数据
    # 示例返回
    return {
        "user_id": user_id,
        "period_days": days,
        "metrics": {
            "avg_score_improvement": 12.5,
            "total_training_time": 25.5,
            "sessions_count": 15,
            "best_score": 92.3,
            "consistency_rating": 0.85
        },
        "daily_scores": [
            {"date": "2024-01-20", "score": 85.5},
            {"date": "2024-01-21", "score": 87.2}
        ]
    }


# 错误处理
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Resource not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    # 创建必要的目录
    Path("uploads").mkdir(exist_ok=True)
    Path("recordings").mkdir(exist_ok=True)
    print("Sports Analyzer API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行"""
    # 清理所有活动会话
    for session_id in list(active_sessions.keys()):
        analyzer = active_sessions[session_id]
        analyzer.stop_analysis()
    active_sessions.clear()
    print("Sports Analyzer API shutdown complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)