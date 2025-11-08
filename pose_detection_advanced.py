# -*- coding: utf-8 -*-
"""
羽毛球视频动作骨架识别程序 - 增强版
支持多种姿态估计模型对比，并检测球、球拍等物体
"""

import sys
import io
import os

# 设置Windows控制台输出编码为UTF-8
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass
    try:
        os.system('chcp 65001 > nul 2>&1')
    except Exception:
        pass

import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import Tuple, List, Optional, Dict
import time


class PoseDetectorBase:
    """姿态检测器基类"""
    
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
        (5, 6),  # 肩膀
        (5, 7), (7, 9),  # 左臂
        (6, 8), (8, 10),  # 右臂
        (11, 12),  # 臀部
        (5, 11), (6, 12),  # 躯干
        (11, 13), (13, 15),  # 左腿
        (12, 14), (14, 16),  # 右腿
    ]
    
    KEYPOINT_NAMES = [
        "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
        "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
        "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
        "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
    ]
    
    def __init__(self, confidence: float = 0.25):
        self.confidence = confidence
        self.model_name = "Base"
    
    def detect(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """检测姿态，返回关键点和置信度"""
        raise NotImplementedError
    
    def draw_skeleton(self, image: np.ndarray, keypoints: np.ndarray,
                     confidences: np.ndarray, show_labels: bool = False) -> np.ndarray:
        """绘制骨架"""
        result_image = image.copy()
        colors = [
            (255, 0, 0),    # 红色 - 头部
            (0, 255, 0),    # 绿色 - 手臂
            (0, 0, 255),    # 蓝色 - 躯干
            (255, 255, 0),  # 青色 - 腿部
        ]
        
        for person_idx, (kp, conf) in enumerate(zip(keypoints, confidences)):
            # 绘制关键点
            for i, (point, conf_score) in enumerate(zip(kp, conf)):
                if conf_score > 0.5:
                    x, y = int(point[0]), int(point[1])
                    if i < 5:
                        color = colors[0]
                        radius = 3  # 头部关键点（较小）
                    elif i < 11:
                        color = colors[1]
                        radius = 3  # 手臂关键点（较小）
                    elif i < 13:
                        color = colors[2]
                        radius = 3  # 躯干关键点（较小）
                    else:
                        color = colors[3]
                        radius = 3  # 腿部关键点（较小）
                    
                    cv2.circle(result_image, (x, y), radius, color, -1)
                    
                    if show_labels:
                        cv2.putText(result_image, self.KEYPOINT_NAMES[i],
                                  (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.4, color, 1)
            
            # 绘制骨架连接（细线条，更优雅）
            for connection in self.SKELETON_CONNECTIONS:
                idx1, idx2 = connection
                if conf[idx1] > 0.5 and conf[idx2] > 0.5:
                    pt1 = (int(kp[idx1][0]), int(kp[idx1][1]))
                    pt2 = (int(kp[idx2][0]), int(kp[idx2][1]))
                    
                    if idx1 < 5 or idx2 < 5:
                        color = colors[0]
                        thickness = 1  # 头部连接（细线）
                    elif idx1 < 11 or idx2 < 11:
                        color = colors[1]
                        thickness = 1  # 手臂连接（细线）
                    elif idx1 < 13 or idx2 < 13:
                        color = colors[2]
                        thickness = 1  # 躯干连接（细线）
                    else:
                        color = colors[3]
                        thickness = 1  # 腿部连接（细线）
                    
                    cv2.line(result_image, pt1, pt2, color, thickness)
        
        return result_image


class YOLOPoseDetector(PoseDetectorBase):
    """YOLO姿态检测器"""
    
    def __init__(self, model_path: str = "yolov8n-pose.pt", confidence: float = 0.25):
        super().__init__(confidence)
        try:
            from ultralytics import YOLO
            print(f"Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)
            self.model_name = f"YOLO-{Path(model_path).stem}"
        except ImportError:
            raise ImportError("ultralytics package is required. Install with: pip install ultralytics")
    
    def detect(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        results = self.model(frame, conf=self.confidence, verbose=False)
        keypoints_list = []
        confidences_list = []
        
        for result in results:
            if result.keypoints is not None and result.keypoints.data is not None:
                for kp_data in result.keypoints.data:
                    kp = kp_data[:, :2].cpu().numpy()
                    conf = kp_data[:, 2].cpu().numpy()
                    keypoints_list.append(kp)
                    confidences_list.append(conf)
        
        return keypoints_list, confidences_list


class MediaPipePoseDetector(PoseDetectorBase):
    """MediaPipe姿态检测器"""
    
    def __init__(self, confidence: float = 0.5):
        super().__init__(confidence)
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=confidence,
                min_tracking_confidence=0.5
            )
            self.model_name = "MediaPipe"
        except ImportError:
            raise ImportError("mediapipe package is required. Install with: pip install mediapipe")
    
    def detect(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        keypoints_list = []
        confidences_list = []
        
        if results.pose_landmarks:
            kp = np.zeros((33, 2))
            conf = np.zeros(33)
            
            h, w = frame.shape[:2]
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                kp[idx] = [landmark.x * w, landmark.y * h]
                conf[idx] = landmark.visibility
            
            # MediaPipe有33个关键点，转换为17个COCO格式
            # 映射MediaPipe到COCO格式（简化版）
            coco_indices = [0, 2, 5, 7, 9, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
            if len(coco_indices) == 17:
                kp_coco = kp[coco_indices]
                conf_coco = conf[coco_indices]
                keypoints_list.append(kp_coco)
                confidences_list.append(conf_coco)
        
        return keypoints_list, confidences_list


class ObjectDetector:
    """物体检测器（用于检测球、球拍等）"""
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.25, 
                 ball_confidence: float = 0.15):
        try:
            from ultralytics import YOLO
            print(f"Loading object detection model: {model_path}")
            self.model = YOLO(model_path)
            self.confidence = confidence
            self.ball_confidence = ball_confidence  # 球的专门置信度阈值（更低）
            
            # 定义羽毛球相关类别
            self.sports_objects = {
                'racket': ['racket', 'tennis racket', 'sports equipment'],
                'ball': ['sports ball', 'ball', 'tennis ball'],
                'person': ['person']
            }
            
            # 获取模型中的类别名称，查找球类别的ID
            self.ball_class_ids = []
            # 通过一次检测来获取模型类别名称
            try:
                # 创建一个小的测试图像来获取模型类别信息
                test_img = np.zeros((640, 640, 3), dtype=np.uint8)
                test_results = self.model(test_img, conf=0.1, verbose=False)
                if test_results and len(test_results) > 0:
                    result = test_results[0]
                    if hasattr(result, 'names'):
                        for cls_id, cls_name in result.names.items():
                            cls_name_lower = cls_name.lower()
                            if 'ball' in cls_name_lower or 'sports ball' in cls_name_lower:
                                self.ball_class_ids.append(cls_id)
                                print(f"Found ball class: {cls_name} (ID: {cls_id})")
            except Exception as e:
                print(f"Warning: Could not detect ball classes: {e}")
                # 使用默认的球类别ID（COCO数据集中sports ball通常是32）
                self.ball_class_ids = [32]  # sports ball in COCO
        except ImportError:
            raise ImportError("ultralytics package is required. Install with: pip install ultralytics")
    
    def detect(self, frame: np.ndarray, filter_classes: Optional[List[str]] = None) -> List[Dict]:
        """检测物体，返回检测结果列表
        Args:
            frame: 输入图像
            filter_classes: 要保留的类别列表（如['sports ball', 'racket']），None表示保留所有
        """
        # 使用更低的置信度进行检测（球的置信度更低，能检测更多球）
        # 使用球的最低置信度阈值来检测所有物体
        min_confidence = min(self.confidence, self.ball_confidence)
        results = self.model(frame, conf=min_confidence, verbose=False)
        detections = []
        
        # 用于去重（避免重复检测）
        seen_boxes = set()
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = result.names[cls_id].lower()
                    
                    # 获取边界框
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox_key = (int(x1), int(y1), int(x2), int(y2))
                    
                    # 去重：如果已经检测过这个位置，跳过
                    if bbox_key in seen_boxes:
                        continue
                    
                    # 对于球类，使用更低的置信度阈值
                    is_ball = 'ball' in cls_name or 'sports ball' in cls_name
                    if is_ball:
                        # 球类：使用球专门的置信度阈值
                        if conf < self.ball_confidence:
                            continue
                    else:
                        # 其他物体：使用标准置信度阈值
                        if conf < self.confidence:
                            continue
                    
                    # 如果指定了过滤类别，只保留匹配的
                    if filter_classes is not None:
                        if not any(target in cls_name for target in [c.lower() for c in filter_classes]):
                            continue
                    
                    detections.append({
                        'class': result.names[cls_id],  # 保留原始名称
                        'class_id': cls_id,
                        'confidence': conf,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
                    
                    seen_boxes.add(bbox_key)
        
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict], 
                       draw_person: bool = False, person_thickness: int = 1,
                       person_min_confidence: float = 0.5, show_labels: bool = False) -> np.ndarray:
        """绘制检测结果
        Args:
            image: 输入图像
            detections: 检测结果列表
            draw_person: 是否绘制人物框（默认False，不绘制）
            person_thickness: 人物框线宽（如果绘制的话，默认1像素，很细）
            person_min_confidence: 人物检测的最小置信度（默认0.5，即50%）
            show_labels: 是否显示标签（默认False，球拍和球不显示标签）
        """
        result_image = image.copy()
        
        # 定义颜色 - 球拍和球使用更醒目的颜色
        colors = {
            'racket': (255, 0, 255),    # 紫色 - 球拍（突出）
            'ball': (0, 255, 255),      # 黄色 - 球（突出）
            'person': (128, 128, 128),  # 灰色 - 人（不突出）
        }
        
        for det in detections:
            cls_name = det['class'].lower()
            conf = det['confidence']
            x1, y1, x2, y2 = det['bbox']
            
            # 判断是否为运动相关物体
            color = (0, 255, 0)  # 默认绿色
            label = ""
            thickness = 2  # 默认线宽
            font_scale = 0.6  # 默认字体大小
            should_draw = False  # 是否绘制
            
            # 检查是否是球拍或球（绘制框选，但不显示标签）
            if 'racket' in cls_name or 'racquet' in cls_name:
                color = colors.get('racket', (255, 0, 255))
                label = f"Racket: {conf:.2f}" if show_labels else ""
                thickness = 4  # 球拍用粗线（4像素）
                font_scale = 0.8  # 球拍标签字体更大
                should_draw = True
            elif 'ball' in cls_name or 'sports ball' in cls_name:
                color = colors.get('ball', (0, 255, 255))
                label = f"Ball: {conf:.2f}" if show_labels else ""
                thickness = 4  # 球用粗线（4像素）
                font_scale = 0.8  # 球标签字体更大
                should_draw = True
            elif 'person' in cls_name:
                # 人物只标注50%以上的（球场上的其他人不需要标注）
                if not draw_person:
                    continue  # 跳过人物，不绘制
                if conf < person_min_confidence:
                    continue  # 置信度低于70%，跳过
                color = colors.get('person', (128, 128, 128))
                label = f"Person: {conf:.2f}" if show_labels else ""
                thickness = person_thickness  # 人物用细线（1像素）
                font_scale = 0.5  # 人物标签字体较小
                should_draw = True
            
            # 绘制边界框
            if should_draw:
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
                
                # 绘制标签（如果启用且标签不为空）
                if label and show_labels:
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                    # 标签背景
                    cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10),
                                 (x1 + label_size[0], y1), color, -1)
                    # 标签文字
                    cv2.putText(result_image, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        
        return result_image


class BadmintonAnalyzer:
    """羽毛球视频分析器"""
    
    def __init__(self, pose_model: str = "yolo", pose_model_path: str = "yolov8n-pose.pt",
                 object_model_path: str = "yolov8n.pt", confidence: float = 0.25,
                 ball_confidence: float = 0.15):
        """初始化分析器"""
        # 初始化姿态检测器
        if pose_model.lower() == "yolo":
            self.pose_detector = YOLOPoseDetector(pose_model_path, confidence)
        elif pose_model.lower() == "mediapipe":
            self.pose_detector = MediaPipePoseDetector(confidence)
        else:
            raise ValueError(f"Unknown pose model: {pose_model}. Use 'yolo' or 'mediapipe'")
        
        # 初始化物体检测器（球的置信度阈值更低）
        self.object_detector = ObjectDetector(object_model_path, confidence, ball_confidence)
        
        print(f"Initialized with pose model: {self.pose_detector.model_name}")
        print(f"Object detection model: {Path(object_model_path).stem}")
        print(f"Ball detection confidence: {ball_confidence} (lower threshold for better detection)")
    
    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     show_preview: bool = True, show_labels: bool = False,
                     save_frames: bool = False, detect_objects: bool = True) -> None:
        """处理视频"""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height}, {fps}fps, Total frames: {total_frames}")
        
        # 设置输出视频
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"Output video will be saved to: {output_path}")
        
        # 创建关键帧保存目录
        frames_dir = None
        if save_frames:
            frames_dir = video_path.parent / f"{video_path.stem}_keyframes"
            frames_dir.mkdir(exist_ok=True)
            print(f"Keyframes will be saved to: {frames_dir}")
        
        frame_count = 0
        keyframe_count = 0
        last_person_count = 0
        total_pose_time = 0
        total_object_time = 0
        
        print("Starting video processing...")
        print("Press 'q' to quit preview, Press 's' to save current frame")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 姿态检测
                start_time = time.time()
                keypoints_list, confidences_list = self.pose_detector.detect(frame)
                pose_time = time.time() - start_time
                total_pose_time += pose_time
                
                # 物体检测（检测球拍、球和人物，但人物只标注70%以上的）
                object_detections = []
                if detect_objects:
                    start_time = time.time()
                    # 检测所有物体（包括球拍、球和人物），绘制时会过滤
                    object_detections = self.object_detector.detect(frame)
                    object_time = time.time() - start_time
                    total_object_time += object_time
                
                # 绘制骨架
                if keypoints_list:
                    keypoints_array = np.array(keypoints_list)
                    confidences_array = np.array(confidences_list)
                    frame = self.pose_detector.draw_skeleton(frame, keypoints_array,
                                                            confidences_array, show_labels)
                
                # 绘制物体检测（球拍和球只绘制框选不显示标签，人物只标注70%以上的）
                if object_detections:
                    frame = self.object_detector.draw_detections(frame, object_detections, 
                                                               draw_person=True,  # 启用人物检测
                                                               person_min_confidence=0.7,  # 只标注70%以上的人物
                                                               show_labels=False)  # 球拍和球不显示标签
                
                # 不显示左上角信息（用户要求）
                person_count = len(keypoints_list)
                object_count = len(object_detections)
                
                # 保存关键帧
                if save_frames and person_count != last_person_count:
                    keyframe_path = frames_dir / f"keyframe_{keyframe_count:04d}_frame_{frame_count}.jpg"
                    cv2.imwrite(str(keyframe_path), frame)
                    keyframe_count += 1
                    print(f"Saved keyframe: {keyframe_path}")
                
                last_person_count = person_count
                
                # 保存到输出视频
                if out:
                    out.write(frame)
                
                # 显示预览
                if show_preview:
                    display_frame = frame.copy()
                    if width > 1280:
                        scale = 1280 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        display_frame = cv2.resize(display_frame, (new_width, new_height))
                    
                    cv2.imshow('Badminton Analysis', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("User interrupted processing")
                        break
                    elif key == ord('s'):
                        if frames_dir:
                            save_path = frames_dir / f"manual_frame_{frame_count:06d}.jpg"
                            cv2.imwrite(str(save_path), frame)
                            print(f"Manually saved frame: {save_path}")
                
                # 显示进度
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    avg_pose_fps = frame_count / total_pose_time if total_pose_time > 0 else 0
                    avg_object_fps = frame_count / total_object_time if total_object_time > 0 else 0
                    print(f"Processing progress: {progress:.1f}% ({frame_count}/{total_frames}) | "
                          f"Avg Pose FPS: {avg_pose_fps:.1f} | Avg Object FPS: {avg_object_fps:.1f}")
        
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            # 如果保存了视频，需要合并音频
            if output_path and Path(output_path).exists():
                print(f"Video saved (without audio): {output_path}")
                print("Merging audio from original video...")
                try:
                    self._merge_audio(str(video_path), str(output_path))
                    print(f"Output video with audio saved: {output_path}")
                except Exception as e:
                    print(f"Warning: Could not merge audio: {e}")
                    print("Please install ffmpeg or use moviepy to merge audio manually")
            
            print(f"\nProcessing completed!")
            print(f"Total frames: {frame_count}")
            if save_frames:
                print(f"Saved keyframes: {keyframe_count}")
            if output_path:
                print(f"Output video saved to: {output_path}")
            if total_pose_time > 0:
                print(f"Average pose detection FPS: {frame_count / total_pose_time:.2f}")
            if total_object_time > 0:
                print(f"Average object detection FPS: {frame_count / total_object_time:.2f}")
    
    def _merge_audio(self, input_video: str, output_video: str) -> None:
        """合并原视频的音频到输出视频"""
        import subprocess
        import tempfile
        
        # 创建临时文件
        temp_video = output_video.replace('.mp4', '_temp.mp4')
        
        # 使用ffmpeg合并音频
        # ffmpeg -i output_video.mp4 -i input_video.mp4 -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest output.mp4
        try:
            cmd = [
                'ffmpeg',
                '-y',  # 覆盖输出文件
                '-i', output_video,  # 输入视频（无音频）
                '-i', input_video,   # 原视频（有音频）
                '-c:v', 'copy',      # 视频编码：直接复制
                '-c:a', 'aac',       # 音频编码：使用AAC
                '-map', '0:v:0',     # 使用第一个输入的视频流
                '-map', '1:a:0',     # 使用第二个输入的音频流
                '-shortest',         # 以最短的流为准
                temp_video
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # 替换原文件
            import shutil
            shutil.move(temp_video, output_video)
            
        except subprocess.CalledProcessError as e:
            # ffmpeg失败，尝试使用moviepy
            try:
                from moviepy.editor import VideoFileClip
                
                # 加载视频和音频
                video_clip = VideoFileClip(output_video)
                audio_clip = VideoFileClip(input_video).audio
                
                # 合并音频
                final_clip = video_clip.set_audio(audio_clip)
                
                # 写入新文件
                final_clip.write_videofile(output_video, codec='libx264', audio_codec='aac', 
                                         temp_audiofile=temp_video.replace('.mp4', '.m4a'),
                                         remove_temp=True)
                
                # 清理
                video_clip.close()
                audio_clip.close()
                final_clip.close()
                
            except ImportError:
                raise ImportError("Please install ffmpeg or moviepy to merge audio. "
                                "Install moviepy: pip install moviepy")
        except FileNotFoundError:
            # ffmpeg未安装，尝试moviepy
            try:
                from moviepy.editor import VideoFileClip
                
                video_clip = VideoFileClip(output_video)
                audio_clip = VideoFileClip(input_video).audio
                final_clip = video_clip.set_audio(audio_clip)
                final_clip.write_videofile(output_video, codec='libx264', audio_codec='aac',
                                         remove_temp=True)
                video_clip.close()
                audio_clip.close()
                final_clip.close()
                
            except ImportError:
                raise ImportError("ffmpeg not found. Please install ffmpeg or moviepy. "
                                "Install moviepy: pip install moviepy")


def compare_models(video_path: str, models: List[str], output_dir: Optional[str] = None):
    """对比多个模型的效果"""
    print(f"Comparing models: {', '.join(models)}")
    
    results = {}
    for model_name in models:
        print(f"\n--- Testing {model_name} ---")
        try:
            if model_name.startswith("yolo"):
                # 提取模型大小
                model_size = model_name.split("-")[1] if "-" in model_name else "n"
                model_path = f"yolov8{model_size}-pose.pt"
                detector = YOLOPoseDetector(model_path)
            elif model_name == "mediapipe":
                detector = MediaPipePoseDetector()
            else:
                print(f"Unknown model: {model_name}, skipping...")
                continue
            
            # 测试单个帧
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print(f"Failed to read video: {video_path}")
                continue
            
            # 检测并计时
            start_time = time.time()
            keypoints_list, confidences_list = detector.detect(frame)
            detection_time = time.time() - start_time
            
            results[model_name] = {
                'detector': detector,
                'detection_time': detection_time,
                'fps': 1.0 / detection_time if detection_time > 0 else 0,
                'persons_detected': len(keypoints_list)
            }
            
            print(f"{model_name}: {detection_time*1000:.2f}ms ({results[model_name]['fps']:.1f} FPS), "
                  f"Detected {len(keypoints_list)} person(s)")
        
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            continue
    
    # 显示对比结果
    print("\n=== Model Comparison Results ===")
    print(f"{'Model':<20} {'Time (ms)':<15} {'FPS':<10} {'Persons':<10}")
    print("-" * 60)
    for model_name, result in results.items():
        print(f"{model_name:<20} {result['detection_time']*1000:<15.2f} "
              f"{result['fps']:<10.1f} {result['persons_detected']:<10}")
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Badminton Video Analysis - Advanced Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with YOLO (nano model, fastest)
  python pose_detection_advanced.py video.mp4
  
  # Use different YOLO model sizes
  python pose_detection_advanced.py video.mp4 --pose-model-size m  # medium (more accurate)
  python pose_detection_advanced.py video.mp4 --pose-model-size l  # large (very accurate)
  
  # Use MediaPipe for pose detection
  python pose_detection_advanced.py video.mp4 --pose-model mediapipe
  
  # Use custom model path
  python pose_detection_advanced.py video.mp4 --pose-model-path custom_model.pt
  
  # Improve ball detection (lower confidence threshold)
  python pose_detection_advanced.py video.mp4 --ball-confidence 0.1  # Lower = more detections
  
  # Use larger object detection model for better accuracy
  python pose_detection_advanced.py video.mp4 --object-model-size m
  
  # Combine: larger model + lower ball confidence
  python pose_detection_advanced.py video.mp4 --object-model-size m --ball-confidence 0.1
  
  # Compare multiple models
  python pose_detection_advanced.py video.mp4 --compare yolo-n,yolo-m,mediapipe
  
  # Disable object detection (only pose)
  python pose_detection_advanced.py video.mp4 --no-objects
        """
    )
    
    parser.add_argument('video', type=str, help='Input video file path')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output video file path')
    parser.add_argument('--pose-model', type=str, default='yolo',
                       choices=['yolo', 'mediapipe'],
                       help='Pose detection model type (default: yolo)')
    parser.add_argument('--pose-model-size', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model size: n(nano), s(small), m(medium), l(large), x(extra large) (default: n)')
    parser.add_argument('--pose-model-path', type=str, default=None,
                       help='Custom path to pose model (overrides --pose-model-size)')
    parser.add_argument('--object-model', type=str, default='yolov8n.pt',
                       help='Object detection model path (default: yolov8n.pt)')
    parser.add_argument('--object-model-size', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Object detection model size: n(nano), s(small), m(medium), l(large), x(extra large) (default: n)')
    parser.add_argument('-c', '--confidence', type=float, default=0.25,
                       help='Detection confidence threshold for general objects (default: 0.25)')
    parser.add_argument('--ball-confidence', type=float, default=0.15,
                       help='Ball detection confidence threshold (lower for better detection, default: 0.15)')
    parser.add_argument('--no-preview', action='store_true',
                       help='Do not show preview window')
    parser.add_argument('--show-labels', action='store_true',
                       help='Show keypoint labels')
    parser.add_argument('-s', '--save-frames', action='store_true',
                       help='Save keyframes when person count changes')
    parser.add_argument('--no-objects', action='store_true',
                       help='Disable object detection (ball, racket)')
    parser.add_argument('--compare', type=str, default=None,
                       help='Compare multiple models (comma-separated: yolo-n,yolo-m,mediapipe)')
    
    args = parser.parse_args()
    
    # 如果指定了对比模式
    if args.compare:
        models = [m.strip() for m in args.compare.split(',')]
        compare_models(args.video, models)
        return
    
    # 设置默认输出路径
    if args.output is None:
        video_path = Path(args.video)
        args.output = str(video_path.parent / f"{video_path.stem}_analysis.mp4")
    
    # 确定姿态模型路径
    pose_model_path = args.pose_model_path
    if pose_model_path is None and args.pose_model == 'yolo':
        # 使用模型大小参数
        pose_model_path = f"yolov8{args.pose_model_size}-pose.pt"
    
    # 确定物体检测模型路径
    object_model_path = args.object_model
    if object_model_path == 'yolov8n.pt' and hasattr(args, 'object_model_size'):
        # 如果使用默认模型且指定了模型大小，使用模型大小参数
        object_model_path = f"yolov8{args.object_model_size}.pt"
    
    # 创建分析器
    try:
        analyzer = BadmintonAnalyzer(
            pose_model=args.pose_model,
            pose_model_path=pose_model_path or 'yolov8n-pose.pt',
            object_model_path=object_model_path,
            confidence=args.confidence,
            ball_confidence=args.ball_confidence
        )
        
        # 处理视频
        analyzer.process_video(
            video_path=args.video,
            output_path=args.output,
            show_preview=not args.no_preview,
            show_labels=args.show_labels,
            save_frames=args.save_frames,
            detect_objects=not args.no_objects
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

