# -*- coding: utf-8 -*-
"""
MediaPipe 姿态检测器
提供33关键点的高精度单人姿态检测，并映射到COCO 17关键点格式以兼容现有系统
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, List, Optional

try:
    from skeleton_renderer import SkeletonRenderer
    USE_ADVANCED_RENDERER = True
except ImportError:
    USE_ADVANCED_RENDERER = False
    print("警告: skeleton_renderer未找到，使用基础绘制")


class MediaPipePoseDetector:
    """MediaPipe 人体姿态检测器（33关键点）"""

    # MediaPipe 33关键点到COCO 17关键点的映射
    # MediaPipe索引 → COCO索引
    MEDIAPIPE_TO_COCO_MAPPING = {
        0: 0,   # 鼻子 → 鼻子
        2: 1,   # 左眼内角 → 左眼
        5: 2,   # 右眼内角 → 右眼
        7: 3,   # 左耳 → 左耳
        8: 4,   # 右耳 → 右耳
        11: 5,  # 左肩 → 左肩
        12: 6,  # 右肩 → 右肩
        13: 7,  # 左肘 → 左肘
        14: 8,  # 右肘 → 右肘
        15: 9,  # 左腕 → 左腕
        16: 10, # 右腕 → 右腕
        23: 11, # 左髋 → 左髋
        24: 12, # 右髋 → 右髋
        25: 13, # 左膝 → 左膝
        26: 14, # 右膝 → 右膝
        27: 15, # 左踝 → 左踝
        28: 16, # 右踝 → 右踝
    }

    # 骨架连接关系（COCO格式）
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

    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 model_complexity: int = 1,
                 line_style: str = "openpose"):
        """
        初始化 MediaPipe 姿态检测器

        Args:
            min_detection_confidence: 最小检测置信度 (0.0-1.0)
            min_tracking_confidence: 最小追踪置信度 (0.0-1.0)
            model_complexity: 模型复杂度 (0=Lite, 1=Full, 2=Heavy)
                             0: 最快，适用于移动设备
                             1: 平衡速度和精度（推荐）
                             2: 最高精度，适用于高端GPU
            line_style: 线条样式 ("uniform", "confidence", "body_part", "openpose", "thick_to_thin")
        """
        print(f"初始化 MediaPipe Pose 检测器...")
        print(f"  - 检测置信度: {min_detection_confidence}")
        print(f"  - 追踪置信度: {min_tracking_confidence}")
        print(f"  - 模型复杂度: {model_complexity} (0=Lite, 1=Full, 2=Heavy)")
        print(f"  - 线条样式: {line_style}")

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 初始化姿态检测器
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # 视频模式（启用追踪优化）
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            enable_segmentation=False  # 不需要分割掩码
        )

        self.confidence = min_detection_confidence

        # 初始化高级渲染器
        if USE_ADVANCED_RENDERER:
            self.renderer = SkeletonRenderer(line_style=line_style, base_line_width=2)
        else:
            self.renderer = None

    def detect(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        检测单人姿态

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            keypoints: 关键点坐标 (1, 17, 2) - COCO格式
            confidences: 关键点置信度 (1, 17)
            如果没有检测到人体，返回 (None, None)
        """
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 设置为不可写以提升性能
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)
        image_rgb.flags.writeable = True

        if not results.pose_landmarks:
            return None, None

        # 提取33个关键点
        h, w = image.shape[:2]
        landmarks = results.pose_landmarks.landmark

        # 转换为COCO 17关键点格式
        keypoints_coco = np.zeros((17, 2), dtype=np.float32)
        confidences_coco = np.zeros(17, dtype=np.float32)

        for mp_idx, coco_idx in self.MEDIAPIPE_TO_COCO_MAPPING.items():
            landmark = landmarks[mp_idx]
            keypoints_coco[coco_idx] = [landmark.x * w, landmark.y * h]
            confidences_coco[coco_idx] = landmark.visibility

        # 添加batch维度以兼容现有代码
        keypoints = np.expand_dims(keypoints_coco, axis=0)  # (1, 17, 2)
        confidences = np.expand_dims(confidences_coco, axis=0)  # (1, 17)

        return keypoints, confidences

    def draw_skeleton(self, image: np.ndarray, keypoints: np.ndarray,
                     confidences: np.ndarray, show_labels: bool = False) -> np.ndarray:
        """
        在图像上绘制骨架（与YOLOv8兼容的接口）

        Args:
            image: 输入图像
            keypoints: 关键点坐标 (N, 17, 2)
            confidences: 关键点置信度 (N, 17)
            show_labels: 是否显示关键点标签

        Returns:
            绘制后的图像
        """
        # 使用高级渲染器（如果可用）
        if self.renderer:
            colors = {
                'head': (255, 0, 0),    # 红色
                'arms': (0, 255, 0),    # 绿色
                'torso': (0, 0, 255),   # 蓝色
                'legs': (255, 255, 0),  # 黄色
            }
            return self.renderer.draw_skeleton(
                image, keypoints, confidences, colors,
                show_keypoints=True, show_labels=show_labels,
                keypoint_names=self.KEYPOINT_NAMES if show_labels else None
            )

        # 降级：使用基础绘制
        result_image = image.copy()

        # 定义颜色（与YOLOv8版本保持一致）
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

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'pose'):
            self.pose.close()


# 测试代码
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python mediapipe_detector.py <视频文件路径>")
        sys.exit(1)

    video_path = sys.argv[1]

    # 创建检测器
    detector = MediaPipePoseDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1  # 平衡模式
    )

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n视频信息:")
    print(f"  - FPS: {fps}")
    print(f"  - 总帧数: {total_frames}")
    print(f"\n按 'q' 退出, 按 's' 保存当前帧\n")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 检测姿态
        keypoints, confidences = detector.detect(frame)

        # 绘制结果
        if keypoints is not None:
            result_frame = detector.draw_skeleton(frame, keypoints, confidences, show_labels=True)

            # 显示信息
            info_text = f"Frame: {frame_count}/{total_frames} | MediaPipe Pose | 33->17 keypoints"
            cv2.putText(result_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            result_frame = frame.copy()
            cv2.putText(result_frame, "No person detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 显示
        cv2.imshow('MediaPipe Pose Detection', result_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_path = f"mediapipe_frame_{frame_count}.jpg"
            cv2.imwrite(save_path, result_frame)
            print(f"保存帧到: {save_path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n处理完成！共处理 {frame_count} 帧")
