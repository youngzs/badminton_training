"""
视频处理模块
处理实时视频流、文件上传和视频分析
"""

import cv2
import numpy as np
import asyncio
import time
from typing import Optional, Callable, Any, Dict, List
from dataclasses import dataclass
from enum import Enum
import threading
from queue import Queue
import base64
from pathlib import Path


class VideoSource(Enum):
    WEBCAM = "webcam"
    FILE = "file"
    STREAM = "stream"


@dataclass
class VideoConfig:
    source_type: VideoSource
    source_path: Optional[str] = None
    fps: int = 30
    resolution: tuple = (1280, 720)
    buffer_size: int = 10
    enable_recording: bool = False
    recording_path: Optional[str] = None


@dataclass
class ProcessedFrame:
    frame_id: int
    timestamp: float
    original_frame: np.ndarray
    processed_frame: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None


class VideoProcessor:
    """视频处理器"""
    
    def __init__(self, config: VideoConfig):
        self.config = config
        self.is_running = False
        self.capture = None
        self.frame_queue = Queue(maxsize=config.buffer_size)
        self.result_queue = Queue(maxsize=config.buffer_size)
        self.frame_count = 0
        self.start_time = None
        self.recorder = None
        self.processing_callbacks: List[Callable] = []
        
    def initialize(self) -> bool:
        """初始化视频源"""
        try:
            if self.config.source_type == VideoSource.WEBCAM:
                self.capture = cv2.VideoCapture(0)
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
                self.capture.set(cv2.CAP_PROP_FPS, self.config.fps)
                
            elif self.config.source_type == VideoSource.FILE:
                if not self.config.source_path:
                    raise ValueError("File path is required for file source")
                self.capture = cv2.VideoCapture(self.config.source_path)
                
            elif self.config.source_type == VideoSource.STREAM:
                if not self.config.source_path:
                    raise ValueError("Stream URL is required for stream source")
                self.capture = cv2.VideoCapture(self.config.source_path)
                
            if not self.capture or not self.capture.isOpened():
                raise RuntimeError("Failed to open video source")
                
            # 初始化录制器
            if self.config.enable_recording and self.config.recording_path:
                self._init_recorder()
                
            return True
            
        except Exception as e:
            print(f"Error initializing video source: {e}")
            return False
    
    def _init_recorder(self):
        """初始化视频录制器"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.recorder = cv2.VideoWriter(
            self.config.recording_path,
            fourcc,
            self.config.fps,
            self.config.resolution
        )
    
    def start(self):
        """启动视频处理"""
        if self.is_running:
            return
            
        self.is_running = True
        self.start_time = time.time()
        
        # 启动帧捕获线程
        capture_thread = threading.Thread(target=self._capture_frames)
        capture_thread.start()
        
        # 启动帧处理线程
        process_thread = threading.Thread(target=self._process_frames)
        process_thread.start()
    
    def stop(self):
        """停止视频处理"""
        self.is_running = False
        
        if self.capture:
            self.capture.release()
            
        if self.recorder:
            self.recorder.release()
            
        cv2.destroyAllWindows()
    
    def _capture_frames(self):
        """捕获视频帧"""
        while self.is_running:
            if self.capture and self.capture.isOpened():
                ret, frame = self.capture.read()
                
                if ret:
                    timestamp = time.time() - self.start_time
                    processed_frame = ProcessedFrame(
                        frame_id=self.frame_count,
                        timestamp=timestamp,
                        original_frame=frame
                    )
                    
                    # 非阻塞添加到队列
                    if not self.frame_queue.full():
                        self.frame_queue.put(processed_frame)
                        self.frame_count += 1
                        
                    # 录制原始帧
                    if self.recorder:
                        self.recorder.write(frame)
                else:
                    # 视频结束或出错
                    if self.config.source_type == VideoSource.FILE:
                        self.is_running = False
                        break
                        
            time.sleep(1.0 / self.config.fps)  # 控制帧率
    
    def _process_frames(self):
        """处理视频帧"""
        while self.is_running or not self.frame_queue.empty():
            if not self.frame_queue.empty():
                frame_data = self.frame_queue.get()
                
                # 执行所有注册的处理回调
                for callback in self.processing_callbacks:
                    try:
                        frame_data = callback(frame_data)
                    except Exception as e:
                        print(f"Error in processing callback: {e}")
                
                # 将处理结果放入结果队列
                if not self.result_queue.full():
                    self.result_queue.put(frame_data)
            else:
                time.sleep(0.01)  # 避免CPU占用过高
    
    def add_processor(self, callback: Callable[[ProcessedFrame], ProcessedFrame]):
        """添加帧处理器"""
        self.processing_callbacks.append(callback)
    
    def get_processed_frame(self) -> Optional[ProcessedFrame]:
        """获取处理后的帧"""
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None
    
    def get_frame_for_display(self) -> Optional[np.ndarray]:
        """获取用于显示的帧"""
        frame_data = self.get_processed_frame()
        if frame_data:
            if frame_data.processed_frame is not None:
                return frame_data.processed_frame
            return frame_data.original_frame
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        if not self.start_time:
            return {}
            
        elapsed_time = time.time() - self.start_time
        actual_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        return {
            "frame_count": self.frame_count,
            "elapsed_time": elapsed_time,
            "actual_fps": actual_fps,
            "target_fps": self.config.fps,
            "queue_size": self.frame_queue.qsize(),
            "result_queue_size": self.result_queue.qsize()
        }
    
    def save_snapshot(self, frame: np.ndarray, path: str):
        """保存截图"""
        cv2.imwrite(path, frame)
    
    def __del__(self):
        """清理资源"""
        self.stop()


class StreamingServer:
    """视频流服务器（用于Web端展示）"""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.clients: List[asyncio.Queue] = []
        self.is_running = False
        
    async def handle_client(self, websocket, path):
        """处理客户端连接"""
        queue = asyncio.Queue()
        self.clients.append(queue)
        
        try:
            while True:
                frame_data = await queue.get()
                await websocket.send(frame_data)
        except Exception as e:
            print(f"Client disconnected: {e}")
        finally:
            self.clients.remove(queue)
    
    async def broadcast_frame(self, frame: np.ndarray):
        """广播视频帧到所有客户端"""
        if not self.clients:
            return
            
        # 编码为JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = base64.b64encode(buffer).decode('utf-8')
        
        # 发送到所有客户端
        for queue in self.clients:
            try:
                await queue.put(frame_data)
            except:
                pass
    
    async def start(self):
        """启动流服务器"""
        import websockets
        self.is_running = True
        await websockets.serve(self.handle_client, "localhost", self.port)
        print(f"Streaming server started on port {self.port}")
    
    def stop(self):
        """停止流服务器"""
        self.is_running = False


class VideoAnalyzer:
    """视频分析器（整合视频处理和动作分析）"""
    
    def __init__(self):
        self.video_processor = None
        self.motion_analyzer = None
        self.feedback_system = None
        self.streaming_server = None
        self.analysis_results = []
        
    def setup(self, video_config: VideoConfig, enable_streaming: bool = False):
        """设置分析器"""
        from .motion_analyzer import MotionAnalyzer, SportType
        from .feedback_system import FeedbackSystem
        
        # 初始化组件
        self.video_processor = VideoProcessor(video_config)
        self.motion_analyzer = MotionAnalyzer(SportType.BADMINTON)
        self.feedback_system = FeedbackSystem(SportType.BADMINTON)
        
        if enable_streaming:
            self.streaming_server = StreamingServer()
        
        # 添加动作分析处理器
        self.video_processor.add_processor(self._analyze_motion)
        
        # 初始化视频源
        return self.video_processor.initialize()
    
    def _analyze_motion(self, frame_data: ProcessedFrame) -> ProcessedFrame:
        """分析动作"""
        # 使用动作分析器处理帧
        motion_frame = self.motion_analyzer.process_frame(
            frame_data.original_frame,
            frame_data.timestamp
        )
        
        if motion_frame:
            # 绘制骨架
            annotated_frame = self.motion_analyzer.draw_skeleton(
                frame_data.original_frame,
                motion_frame
            )
            frame_data.processed_frame = annotated_frame
            
            # 保存分析结果
            frame_data.metadata = {
                "score": motion_frame.score,
                "issues": motion_frame.issues,
                "joint_angles": motion_frame.joint_angles
            }
            
            self.analysis_results.append(motion_frame)
            
            # 限制结果缓存大小
            if len(self.analysis_results) > 300:
                self.analysis_results.pop(0)
        
        return frame_data
    
    async def start_analysis(self):
        """开始分析"""
        self.video_processor.start()
        
        if self.streaming_server:
            await self.streaming_server.start()
        
        # 主循环
        while self.video_processor.is_running:
            frame = self.video_processor.get_frame_for_display()
            
            if frame is not None:
                # 显示本地窗口
                cv2.imshow("Sports Analyzer", frame)
                
                # 流式传输
                if self.streaming_server:
                    await self.streaming_server.broadcast_frame(frame)
                
                # ESC键退出
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            
            await asyncio.sleep(0.01)
    
    def stop_analysis(self):
        """停止分析"""
        self.video_processor.stop()
        
        if self.streaming_server:
            self.streaming_server.stop()
        
        # 生成最终反馈
        if self.analysis_results:
            feedback = self.feedback_system.generate_feedback(self.analysis_results)
            return feedback
        
        return None
    
    def get_current_stats(self) -> Dict[str, Any]:
        """获取当前统计信息"""
        stats = self.video_processor.get_stats()
        
        if self.analysis_results:
            recent_scores = [r.score for r in self.analysis_results[-30:]]
            stats["avg_score"] = np.mean(recent_scores)
            stats["current_score"] = self.analysis_results[-1].score
            
        return stats


# 使用示例
if __name__ == "__main__":
    # 配置视频源
    config = VideoConfig(
        source_type=VideoSource.WEBCAM,
        fps=30,
        resolution=(1280, 720),
        enable_recording=True,
        recording_path="training_session.mp4"
    )
    
    # 创建分析器
    analyzer = VideoAnalyzer()
    
    # 设置并启动
    if analyzer.setup(config, enable_streaming=True):
        # 运行异步分析
        asyncio.run(analyzer.start_analysis())
        
        # 获取反馈
        feedback = analyzer.stop_analysis()
        if feedback:
            print(f"Training Score: {feedback.overall_score:.2f}")
            print(f"Level: {feedback.level.value}")
            print(f"Suggestions: {[s.title for s in feedback.suggestions]}")