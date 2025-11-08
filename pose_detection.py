# -*- coding: utf-8 -*-
"""
羽毛球视频动作骨架识别程序
使用YOLOv8进行人体姿态估计，绘制动作骨架用于教学讲解
"""

import sys
import io
import os

# 设置Windows控制台输出编码为UTF-8
def safe_print(text):
    """安全输出中文文本"""
    try:
        if sys.platform == 'win32':
            # 尝试使用UTF-8编码输出
            sys.stdout.buffer.write(text.encode('utf-8'))
            sys.stdout.buffer.write(b'\n')
            sys.stdout.buffer.flush()
        else:
            print(text)
    except Exception:
        # 如果失败，使用默认print
        print(text)

if sys.platform == 'win32':
    try:
        # Python 3.7+
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python 3.6及以下版本
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass
    # 设置控制台代码页为UTF-8
    try:
        os.system('chcp 65001 > nul 2>&1')
    except Exception:
        pass

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse
from typing import Tuple, List, Optional


class PoseDetector:
    """人体姿态检测器"""
    
    # 骨架连接关系（COCO格式的关键点）
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
    
    # 关键点名称（COCO格式）
    KEYPOINT_NAMES = [
        "鼻子", "左眼", "右眼", "左耳", "右耳",
        "左肩", "右肩", "左肘", "右肘", "左腕", "右腕",
        "左髋", "右髋", "左膝", "右膝", "左踝", "右踝"
    ]
    
    def __init__(self, model_path: str = "yolov8n-pose.pt", confidence: float = 0.25):
        """
        初始化姿态检测器
        
        Args:
            model_path: YOLO模型路径，默认使用yolov8n-pose.pt
            confidence: 检测置信度阈值
        """
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.confidence = confidence
        
    def draw_skeleton(self, image: np.ndarray, keypoints: np.ndarray, 
                     confidences: np.ndarray, show_labels: bool = False) -> np.ndarray:
        """
        在图像上绘制骨架
        
        Args:
            image: 输入图像
            keypoints: 关键点坐标 (N, 17, 2)
            confidences: 关键点置信度 (N, 17)
            show_labels: 是否显示关键点标签
            
        Returns:
            绘制后的图像
        """
        result_image = image.copy()
        
        # 定义颜色
        colors = [
            (255, 0, 0),    # 红色 - 头部
            (0, 255, 0),    # 绿色 - 手臂
            (0, 0, 255),    # 蓝色 - 躯干
            (255, 255, 0),  # 青色 - 腿部
        ]
        
        for person_idx, (kp, conf) in enumerate(zip(keypoints, confidences)):
            # 绘制关键点
            for i, (point, conf_score) in enumerate(zip(kp, conf)):
                if conf_score > 0.5:  # 只绘制置信度高的关键点
                    x, y = int(point[0]), int(point[1])
                    # 根据关键点类型选择颜色
                    if i < 5:  # 头部
                        color = colors[0]
                        radius = 5
                    elif i < 11:  # 手臂
                        color = colors[1]
                        radius = 4
                    elif i < 13:  # 躯干
                        color = colors[2]
                        radius = 4
                    else:  # 腿部
                        color = colors[3]
                        radius = 4
                    
                    cv2.circle(result_image, (x, y), radius, color, -1)
                    
                    # 显示标签
                    if show_labels:
                        cv2.putText(result_image, self.KEYPOINT_NAMES[i], 
                                  (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.4, color, 1)
            
            # 绘制骨架连接
            for connection in self.SKELETON_CONNECTIONS:
                idx1, idx2 = connection
                if conf[idx1] > 0.5 and conf[idx2] > 0.5:
                    pt1 = (int(kp[idx1][0]), int(kp[idx1][1]))
                    pt2 = (int(kp[idx2][0]), int(kp[idx2][1]))
                    
                    # 根据连接类型选择颜色
                    if idx1 < 5 or idx2 < 5:  # 头部连接
                        color = colors[0]
                    elif idx1 < 11 or idx2 < 11:  # 手臂连接
                        color = colors[1]
                    elif idx1 < 13 or idx2 < 13:  # 躯干连接
                        color = colors[2]
                    else:  # 腿部连接
                        color = colors[3]
                    
                    cv2.line(result_image, pt1, pt2, color, 2)
        
        return result_image
    
    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     show_preview: bool = True, show_labels: bool = False,
                     save_frames: bool = False) -> None:
        """
        处理视频文件，检测姿态并绘制骨架
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径，如果为None则不保存
            show_preview: 是否显示预览窗口
            show_labels: 是否显示关键点标签
            save_frames: 是否保存关键帧（用于教学讲解）
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 打开视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
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
        
        print("Starting video processing...")
        print("Press 'q' to quit preview, Press 's' to save current frame")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 使用YOLO进行姿态估计
                results = self.model(frame, conf=self.confidence, verbose=False)
                
                # 提取关键点
                keypoints_list = []
                confidences_list = []
                
                for result in results:
                    if result.keypoints is not None and result.keypoints.data is not None:
                        for kp_data in result.keypoints.data:
                            # kp_data形状: [x1, y1, conf1, x2, y2, conf2, ...]
                            kp = kp_data[:, :2].cpu().numpy()  # 提取坐标
                            conf = kp_data[:, 2].cpu().numpy()  # 提取置信度
                            keypoints_list.append(kp)
                            confidences_list.append(conf)
                
                # 绘制骨架
                if keypoints_list:
                    keypoints_array = np.array(keypoints_list)
                    confidences_array = np.array(confidences_list)
                    frame = self.draw_skeleton(frame, keypoints_array, 
                                              confidences_array, show_labels)
                
                # 在图像上显示信息
                person_count = len(keypoints_list)
                # info_text = [
                #     f"Frame: {frame_count}/{total_frames}",
                #     f"Persons: {person_count}",
                #     f"Press 'q' to quit, 's' to save"
                # ]
                y_offset = 30
                for text in info_text:
                    cv2.putText(frame, text, (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_offset += 30
                
                # 保存关键帧（当检测到的人数发生变化时）
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
                    # 调整显示尺寸（如果太大）
                    display_frame = frame.copy()
                    if width > 1280:
                        scale = 1280 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        display_frame = cv2.resize(display_frame, (new_width, new_height))
                    
                    cv2.imshow('羽毛球动作骨架识别', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("User interrupted processing")
                        break
                    elif key == ord('s'):
                        # 手动保存当前帧
                        if frames_dir:
                            save_path = frames_dir / f"manual_frame_{frame_count:06d}.jpg"
                            cv2.imwrite(str(save_path), frame)
                            print(f"Manually saved frame: {save_path}")
                
                # 显示进度
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Processing progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            print(f"\nProcessing completed!")
            print(f"Total frames: {frame_count}")
            if save_frames:
                print(f"Saved keyframes: {keyframe_count}")
            if output_path:
                print(f"Output video saved to: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='羽毛球视频动作骨架识别程序',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本使用（使用默认模型）
  python pose_detection.py video.mp4
  
  # 指定输出文件
  python pose_detection.py video.mp4 -o output.mp4
  
  # 使用更大的模型（更准确但更慢）
  python pose_detection.py video.mp4 -m yolov8m-pose.pt
  
  # 保存关键帧用于教学
  python pose_detection.py video.mp4 -s
  
  # 显示关键点标签
  python pose_detection.py video.mp4 --show-labels
        """
    )
    
    parser.add_argument('video', type=str, help='输入视频文件路径')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='输出视频文件路径（默认：原文件名_pose.mp4）')
    parser.add_argument('-m', '--model', type=str, default='yolov8n-pose.pt',
                       help='YOLO模型路径（默认：yolov8n-pose.pt）')
    parser.add_argument('-c', '--confidence', type=float, default=0.25,
                       help='检测置信度阈值（默认：0.25）')
    parser.add_argument('--no-preview', action='store_true',
                       help='不显示预览窗口')
    parser.add_argument('--show-labels', action='store_true',
                       help='显示关键点标签')
    parser.add_argument('-s', '--save-frames', action='store_true',
                       help='保存关键帧（当检测到的人数变化时）')
    
    args = parser.parse_args()
    
    # 设置默认输出路径
    if args.output is None:
        video_path = Path(args.video)
        args.output = str(video_path.parent / f"{video_path.stem}_pose.mp4")
    
    # 创建检测器
    detector = PoseDetector(model_path=args.model, confidence=args.confidence)
    
    # 处理视频
    try:
        detector.process_video(
            video_path=args.video,
            output_path=args.output,
            show_preview=not args.no_preview,
            show_labels=args.show_labels,
            save_frames=args.save_frames
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

