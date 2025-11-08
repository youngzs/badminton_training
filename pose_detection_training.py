# -*- coding: utf-8 -*-
"""
羽毛球训练视频分析系统 - GPU加速版
专为教练指导和学员训练设计，支持慢动作回放和AI动作分析
"""

import sys
import io
import os

# 设置Windows控制台输出编码为UTF-8
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import Tuple, List, Optional, Dict
import time
import torch
import json
from collections import deque

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)


class GPUOptimizedPoseDetector:
    """GPU优化的姿态检测器"""

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
        "鼻子", "左眼", "右眼", "左耳", "右耳",
        "左肩", "右肩", "左肘", "右肘", "左腕", "右腕",
        "左髋", "右髋", "左膝", "右膝", "左踝", "右踝"
    ]

    # 关键部位组（用于动作分析）
    BODY_PARTS = {
        'head': [0, 1, 2, 3, 4],
        'left_arm': [5, 7, 9],
        'right_arm': [6, 8, 10],
        'torso': [5, 6, 11, 12],
        'left_leg': [11, 13, 15],
        'right_leg': [12, 14, 16]
    }

    def __init__(self, model_path: str = "yolov8n-pose.pt", confidence: float = 0.25,
                 device: str = "auto", use_fp16: bool = True, batch_size: int = 1):
        """
        初始化GPU优化的姿态检测器

        Args:
            model_path: YOLO模型路径
            confidence: 检测置信度阈值
            device: 设备选择 ('auto', 'cuda', 'cpu')
            use_fp16: 是否使用FP16加速（需要GPU）
            batch_size: 批处理大小（GPU模式下可设置>1）
        """
        print(f"🚀 初始化GPU优化姿态检测器...")

        # 自动选择设备
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"   设备: {self.device}")
        if self.device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        # 加载模型
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # FP16优化（仅GPU）
        self.use_fp16 = use_fp16 and self.device == "cuda"
        if self.use_fp16:
            print(f"   ✓ 启用FP16半精度推理（2x速度提升）")

        self.confidence = confidence
        self.batch_size = batch_size
        self.model_name = f"YOLO-{Path(model_path).stem}"

        # 性能统计
        self.inference_times = deque(maxlen=100)

        print(f"✓ 模型加载完成: {self.model_name}")

    def detect_batch(self, frames: List[np.ndarray]) -> List[Tuple[List[np.ndarray], List[np.ndarray]]]:
        """批量检测（GPU加速）"""
        if len(frames) == 0:
            return []

        start_time = time.time()

        # YOLO批量推理
        results = self.model(frames, conf=self.confidence, verbose=False,
                            half=self.use_fp16, device=self.device)

        # 提取关键点
        batch_results = []
        for result in results:
            keypoints_list = []
            confidences_list = []

            if result.keypoints is not None and result.keypoints.data is not None:
                for kp_data in result.keypoints.data:
                    kp = kp_data[:, :2].cpu().numpy()
                    conf = kp_data[:, 2].cpu().numpy()
                    keypoints_list.append(kp)
                    confidences_list.append(conf)

            batch_results.append((keypoints_list, confidences_list))

        # 记录性能
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)

        return batch_results

    def detect(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """单帧检测"""
        batch_results = self.detect_batch([frame])
        return batch_results[0] if batch_results else ([], [])

    def get_avg_fps(self) -> float:
        """获取平均FPS"""
        if len(self.inference_times) == 0:
            return 0.0
        avg_time = sum(self.inference_times) / len(self.inference_times)
        return self.batch_size / avg_time if avg_time > 0 else 0.0


class PoseAnalyzer:
    """羽毛球动作分析器"""

    def __init__(self):
        """初始化动作分析器"""
        self.reference_poses = {}  # 标准动作库

    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """计算三点之间的角度"""
        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    def analyze_pose(self, keypoints: np.ndarray, confidences: np.ndarray) -> Dict:
        """
        分析单个姿势

        Args:
            keypoints: 单人关键点坐标 shape (17, 2)
            confidences: 单人关键点置信度 shape (17,)

        Returns:
            分析结果字典，包含角度、问题、建议等
        """
        if len(keypoints) == 0:
            return {"valid": False}

        # keypoints 和 confidences 已经是单人数据，不需要再索引
        kp = keypoints
        conf = confidences

        analysis = {
            "valid": True,
            "angles": {},
            "issues": [],
            "suggestions": []
        }

        # 计算关键角度
        try:
            # 右臂角度（肩-肘-腕）
            if all(conf[[6, 8, 10]] > 0.5):
                right_arm_angle = self.calculate_angle(kp[6], kp[8], kp[10])
                analysis["angles"]["right_elbow"] = right_arm_angle

                # 分析建议
                if right_arm_angle < 90:
                    analysis["issues"].append("右臂弯曲不足")
                    analysis["suggestions"].append("击球时右臂应充分伸展")

            # 左臂角度
            if all(conf[[5, 7, 9]] > 0.5):
                left_arm_angle = self.calculate_angle(kp[5], kp[7], kp[9])
                analysis["angles"]["left_elbow"] = left_arm_angle

            # 躯干角度（肩-髋）
            if all(conf[[5, 6, 11, 12]] > 0.5):
                shoulder_center = (kp[5] + kp[6]) / 2
                hip_center = (kp[11] + kp[12]) / 2
                torso_vertical = np.array([0, -1])  # 垂直向上
                torso_vector = shoulder_center - hip_center

                cos_angle = np.dot(torso_vector, torso_vertical) / (np.linalg.norm(torso_vector) + 1e-6)
                torso_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                analysis["angles"]["torso_lean"] = torso_angle

                if torso_angle > 30:
                    analysis["issues"].append("身体前倾过度")
                    analysis["suggestions"].append("保持躯干相对直立")

            # 膝盖弯曲（右腿）
            if all(conf[[12, 14, 16]] > 0.5):
                right_knee_angle = self.calculate_angle(kp[12], kp[14], kp[16])
                analysis["angles"]["right_knee"] = right_knee_angle

                if right_knee_angle < 140:
                    analysis["issues"].append("右膝弯曲较多，重心较低")
                elif right_knee_angle > 170:
                    analysis["issues"].append("右腿过直，缺乏弹性")
                    analysis["suggestions"].append("保持膝盖微曲，便于移动")

        except Exception as e:
            print(f"姿势分析出错: {e}")

        return analysis

    def compare_with_reference(self, current_kp: np.ndarray, reference_kp: np.ndarray) -> Dict:
        """与标准动作对比"""
        # 计算关键点的相对位置差异
        # 归一化到肩宽
        pass

    def load_reference_pose(self, pose_file: str):
        """加载标准动作"""
        try:
            with open(pose_file, 'r') as f:
                self.reference_poses = json.load(f)
            print(f"✓ 加载标准动作库: {len(self.reference_poses)} 个动作")
        except FileNotFoundError:
            print(f"⚠ 标准动作库不存在: {pose_file}")


class SlowMotionProcessor:
    """慢动作处理器（帧插值）"""

    def __init__(self, slowdown_factor: float = 0.5):
        """
        Args:
            slowdown_factor: 慢动作因子（0.5 = 2倍慢放，0.25 = 4倍慢放）
        """
        self.slowdown_factor = slowdown_factor

    def interpolate_frames(self, frame1: np.ndarray, frame2: np.ndarray,
                          num_interpolations: int = 1) -> List[np.ndarray]:
        """
        光流法帧插值

        Args:
            frame1: 第一帧
            frame2: 第二帧
            num_interpolations: 插值帧数

        Returns:
            插值后的帧列表
        """
        interpolated = []

        # 简单线性插值（生产环境可用RIFE/FILM等深度学习模型）
        for i in range(1, num_interpolations + 1):
            alpha = i / (num_interpolations + 1)
            blended = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
            interpolated.append(blended)

        return interpolated


class TrainingVideoAnalyzer:
    """羽毛球训练视频分析系统"""

    def __init__(self, pose_model_path: str = "yolov8n-pose.pt",
                 confidence: float = 0.25,
                 use_gpu: bool = True,
                 use_fp16: bool = True,
                 batch_size: int = 1):
        """
        初始化训练视频分析系统

        Args:
            pose_model_path: 姿态检测模型路径
            confidence: 检测置信度
            use_gpu: 是否使用GPU
            use_fp16: 是否使用FP16（GPU）
            batch_size: 批处理大小
        """
        print("="*60)
        print("羽毛球训练视频分析系统 - GPU加速版")
        print("="*60)

        # 初始化姿态检测器
        device = "cuda" if use_gpu else "cpu"
        self.pose_detector = GPUOptimizedPoseDetector(
            model_path=pose_model_path,
            confidence=confidence,
            device=device,
            use_fp16=use_fp16,
            batch_size=batch_size
        )

        # 初始化动作分析器
        self.pose_analyzer = PoseAnalyzer()

        # 初始化慢动作处理器
        self.slowmo_processor = SlowMotionProcessor()

        print("="*60)

    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     slowmo_factor: float = 1.0,
                     analyze_pose: bool = True,
                     show_analysis: bool = True,
                     save_keyframes: bool = True,
                     show_preview: bool = True) -> None:
        """
        处理训练视频

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            slowmo_factor: 慢动作因子（1.0=正常，0.5=2倍慢放）
            analyze_pose: 是否进行姿势分析
            show_analysis: 是否在视频上显示分析结果
            save_keyframes: 是否保存关键帧
            show_preview: 是否显示预览
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        # 视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_fps = int(fps / slowmo_factor)

        print(f"\n📹 视频信息:")
        print(f"   分辨率: {width}x{height}")
        print(f"   帧率: {fps} fps → {output_fps} fps (慢动作: {1/slowmo_factor:.1f}x)")
        print(f"   总帧数: {total_frames}")

        # 输出设置
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
            print(f"   输出: {output_path}")

        # 关键帧目录
        keyframes_dir = None
        if save_keyframes:
            keyframes_dir = video_path.parent / f"{video_path.stem}_training_keyframes"
            keyframes_dir.mkdir(exist_ok=True)
            print(f"   关键帧: {keyframes_dir}")

        print(f"\n⚡ 开始处理...")

        frame_count = 0
        keyframe_count = 0
        prev_frame = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # 姿态检测
                keypoints_list, confidences_list = self.pose_detector.detect(frame)

                # 姿势分析
                analysis = None
                if analyze_pose and len(keypoints_list) > 0:
                    analysis = self.pose_analyzer.analyze_pose(
                        np.array(keypoints_list),
                        np.array(confidences_list)
                    )

                # 绘制骨架
                annotated_frame = self.draw_skeleton(frame, keypoints_list, confidences_list)

                # 显示分析结果
                if show_analysis and analysis and analysis["valid"]:
                    annotated_frame = self.draw_analysis(annotated_frame, analysis)

                # 保存关键帧（检测到明显动作变化时）
                if save_keyframes and len(keypoints_list) > 0:
                    # 简单规则：每30帧保存一次
                    if frame_count % 30 == 0:
                        keyframe_path = keyframes_dir / f"keyframe_{keyframe_count:04d}_frame_{frame_count:06d}.jpg"
                        cv2.imwrite(str(keyframe_path), annotated_frame)
                        keyframe_count += 1

                # 慢动作处理
                if slowmo_factor < 1.0 and prev_frame is not None:
                    # 插值帧数
                    num_interpolations = int(1 / slowmo_factor) - 1
                    interpolated_frames = self.slowmo_processor.interpolate_frames(
                        prev_frame, annotated_frame, num_interpolations
                    )

                    # 写入插值帧
                    for interp_frame in interpolated_frames:
                        if out:
                            out.write(interp_frame)

                # 写入当前帧
                if out:
                    out.write(annotated_frame)

                prev_frame = annotated_frame.copy()

                # 显示预览
                if show_preview:
                    display_frame = annotated_frame.copy()
                    if width > 1280:
                        scale = 1280 / width
                        new_w, new_h = int(width * scale), int(height * scale)
                        display_frame = cv2.resize(display_frame, (new_w, new_h))

                    cv2.imshow('羽毛球训练分析', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n⏹ 用户中断")
                        break

                # 进度显示
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    avg_fps = self.pose_detector.get_avg_fps()
                    print(f"   进度: {progress:.1f}% ({frame_count}/{total_frames}) | "
                          f"检测FPS: {avg_fps:.1f}")

        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()

            print(f"\n✓ 处理完成!")
            print(f"   处理帧数: {frame_count}")
            if save_keyframes:
                print(f"   保存关键帧: {keyframe_count}")
            print(f"   平均检测FPS: {self.pose_detector.get_avg_fps():.1f}")

    def draw_skeleton(self, image: np.ndarray, keypoints_list: List[np.ndarray],
                     confidences_list: List[np.ndarray]) -> np.ndarray:
        """绘制骨架"""
        result = image.copy()

        colors = [
            (255, 0, 0),    # 红色 - 头部
            (0, 255, 0),    # 绿色 - 手臂
            (0, 0, 255),    # 蓝色 - 躯干
            (255, 255, 0),  # 青色 - 腿部
        ]

        for kp, conf in zip(keypoints_list, confidences_list):
            # 绘制关键点
            for i, (point, conf_score) in enumerate(zip(kp, conf)):
                if conf_score > 0.5:
                    x, y = int(point[0]), int(point[1])
                    if i < 5:
                        color = colors[0]
                    elif i < 11:
                        color = colors[1]
                    elif i < 13:
                        color = colors[2]
                    else:
                        color = colors[3]
                    cv2.circle(result, (x, y), 4, color, -1)

            # 绘制骨架连接
            for idx1, idx2 in self.pose_detector.SKELETON_CONNECTIONS:
                if conf[idx1] > 0.5 and conf[idx2] > 0.5:
                    pt1 = (int(kp[idx1][0]), int(kp[idx1][1]))
                    pt2 = (int(kp[idx2][0]), int(kp[idx2][1]))

                    if idx1 < 5 or idx2 < 5:
                        color = colors[0]
                    elif idx1 < 11 or idx2 < 11:
                        color = colors[1]
                    elif idx1 < 13 or idx2 < 13:
                        color = colors[2]
                    else:
                        color = colors[3]

                    cv2.line(result, pt1, pt2, color, 2)

        return result

    def draw_analysis(self, image: np.ndarray, analysis: Dict) -> np.ndarray:
        """在图像上绘制分析结果"""
        result = image.copy()
        h, w = image.shape[:2]

        # 半透明背景
        overlay = result.copy()
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)

        y_offset = 30

        # 标题
        cv2.putText(result, "动作分析", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += 30

        # 显示角度
        if "angles" in analysis and analysis["angles"]:
            for joint, angle in analysis["angles"].items():
                text = f"{joint}: {angle:.1f}度"
                cv2.putText(result, text, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25

        # 显示问题
        if analysis.get("issues"):
            cv2.putText(result, "问题:", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 25
            for issue in analysis["issues"][:2]:  # 最多显示2个
                cv2.putText(result, f"• {issue}", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

        return result


def main():
    parser = argparse.ArgumentParser(
        description='羽毛球训练视频分析系统 - GPU加速版',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本使用（GPU加速）
  python pose_detection_training.py video.mp4

  # 2倍慢动作回放
  python pose_detection_training.py video.mp4 --slowmo 0.5 -o output_slow.mp4

  # 使用更大模型获得更高精度
  python pose_detection_training.py video.mp4 -m yolov8m-pose.pt

  # 4倍慢动作 + 动作分析
  python pose_detection_training.py video.mp4 --slowmo 0.25 --analyze

  # CPU模式（无GPU时）
  python pose_detection_training.py video.mp4 --no-gpu
        """
    )

    parser.add_argument('video', type=str, help='输入视频文件')
    parser.add_argument('-o', '--output', type=str, help='输出视频文件')
    parser.add_argument('-m', '--model', type=str, default='yolov8n-pose.pt',
                       help='YOLO模型路径 (默认: yolov8n-pose.pt)')
    parser.add_argument('-c', '--confidence', type=float, default=0.25,
                       help='检测置信度 (默认: 0.25)')
    parser.add_argument('--slowmo', type=float, default=1.0,
                       help='慢动作因子 (1.0=正常, 0.5=2倍慢, 0.25=4倍慢)')
    parser.add_argument('--analyze', action='store_true',
                       help='启用姿势分析')
    parser.add_argument('--no-gpu', action='store_true',
                       help='禁用GPU加速')
    parser.add_argument('--no-fp16', action='store_true',
                       help='禁用FP16加速')
    parser.add_argument('--no-preview', action='store_true',
                       help='不显示预览窗口')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='批处理大小 (GPU模式，默认: 1)')

    args = parser.parse_args()

    # 设置默认输出路径
    if args.output is None:
        video_path = Path(args.video)
        suffix = f"_training_slow{int(1/args.slowmo)}x" if args.slowmo < 1.0 else "_training"
        args.output = str(video_path.parent / f"{video_path.stem}{suffix}.mp4")

    try:
        # 创建分析器
        analyzer = TrainingVideoAnalyzer(
            pose_model_path=args.model,
            confidence=args.confidence,
            use_gpu=not args.no_gpu,
            use_fp16=not args.no_fp16,
            batch_size=args.batch_size
        )

        # 处理视频
        analyzer.process_video(
            video_path=args.video,
            output_path=args.output,
            slowmo_factor=args.slowmo,
            analyze_pose=args.analyze,
            show_analysis=True,
            save_keyframes=True,
            show_preview=not args.no_preview
        )

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
