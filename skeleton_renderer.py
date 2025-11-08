# -*- coding: utf-8 -*-
"""
骨架渲染器 - 支持多种线条样式
提供统一、高级的骨架可视化效果
"""

import cv2
import numpy as np
from typing import Tuple, List
from enum import Enum


class LineStyle(Enum):
    """线条样式枚举"""
    UNIFORM = "uniform"              # 统一粗细（当前默认）
    CONFIDENCE = "confidence"        # 置信度驱动
    BODY_PART = "body_part"         # 身体部位分级
    OPENPOSE = "openpose"           # OpenPose风格渐变
    THICK_TO_THIN = "thick_to_thin" # 从躯干到四肢渐细


class SkeletonRenderer:
    """骨架渲染器 - 高级可视化"""

    # COCO格式骨架连接
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

    # 身体部位分组
    HEAD_CONNECTIONS = [(0, 1), (0, 2), (1, 3), (2, 4)]
    TORSO_CONNECTIONS = [(5, 6), (11, 12), (5, 11), (6, 12)]
    ARM_CONNECTIONS = [(5, 7), (7, 9), (6, 8), (8, 10)]
    LEG_CONNECTIONS = [(11, 13), (13, 15), (12, 14), (14, 16)]

    # 默认颜色
    DEFAULT_COLORS = {
        'head': (255, 0, 0),    # 红色
        'arms': (0, 255, 0),    # 绿色
        'torso': (0, 0, 255),   # 蓝色
        'legs': (255, 255, 0),  # 黄色
    }

    def __init__(self, line_style: str = "uniform", base_line_width: int = 2):
        """
        初始化渲染器

        Args:
            line_style: 线条样式 ("uniform", "confidence", "body_part", "openpose", "thick_to_thin")
            base_line_width: 基础线宽（作为参考值）
        """
        self.line_style = LineStyle(line_style)
        self.base_line_width = base_line_width

    def draw_skeleton(self,
                     image: np.ndarray,
                     keypoints: np.ndarray,
                     confidences: np.ndarray,
                     colors: dict = None,
                     show_keypoints: bool = True,
                     show_labels: bool = False,
                     keypoint_names: List[str] = None) -> np.ndarray:
        """
        绘制骨架（高级版本）

        Args:
            image: 输入图像
            keypoints: 关键点坐标 (N, 17, 2)
            confidences: 关键点置信度 (N, 17)
            colors: 自定义颜色字典
            show_keypoints: 是否显示关键点
            show_labels: 是否显示标签
            keypoint_names: 关键点名称列表

        Returns:
            绘制后的图像
        """
        result = image.copy()

        if colors is None:
            colors = self.DEFAULT_COLORS

        for person_idx, (kp, conf) in enumerate(zip(keypoints, confidences)):
            # 绘制骨架连接
            for connection in self.SKELETON_CONNECTIONS:
                idx1, idx2 = connection

                if conf[idx1] > 0.5 and conf[idx2] > 0.5:
                    pt1 = (int(kp[idx1][0]), int(kp[idx1][1]))
                    pt2 = (int(kp[idx2][0]), int(kp[idx2][1]))

                    # 确定颜色
                    color = self._get_connection_color(connection, colors)

                    # 确定线宽
                    line_width = self._get_line_width(
                        connection, conf[idx1], conf[idx2], kp[idx1], kp[idx2]
                    )

                    # 绘制连接（根据样式）
                    if self.line_style == LineStyle.OPENPOSE:
                        self._draw_gradient_line(result, pt1, pt2, color, line_width)
                    else:
                        cv2.line(result, pt1, pt2, color, line_width)

            # 绘制关键点
            if show_keypoints:
                self._draw_keypoints(result, kp, conf, colors)

            # 绘制标签
            if show_labels and keypoint_names:
                self._draw_labels(result, kp, conf, keypoint_names, colors)

        return result

    def _get_connection_color(self, connection: Tuple[int, int], colors: dict) -> Tuple[int, int, int]:
        """获取连接线的颜色"""
        if connection in self.HEAD_CONNECTIONS:
            return colors.get('head', self.DEFAULT_COLORS['head'])
        elif connection in self.TORSO_CONNECTIONS:
            return colors.get('torso', self.DEFAULT_COLORS['torso'])
        elif connection in self.ARM_CONNECTIONS:
            return colors.get('arms', self.DEFAULT_COLORS['arms'])
        else:  # LEG_CONNECTIONS
            return colors.get('legs', self.DEFAULT_COLORS['legs'])

    def _get_line_width(self,
                       connection: Tuple[int, int],
                       conf1: float,
                       conf2: float,
                       pt1: np.ndarray,
                       pt2: np.ndarray) -> int:
        """
        根据样式计算线宽

        Args:
            connection: 连接关系
            conf1, conf2: 两个关键点的置信度
            pt1, pt2: 两个关键点的坐标

        Returns:
            线宽（像素）
        """
        if self.line_style == LineStyle.UNIFORM:
            # 统一粗细
            return self.base_line_width

        elif self.line_style == LineStyle.CONFIDENCE:
            # 置信度驱动：置信度越高，线越粗
            avg_conf = (conf1 + conf2) / 2
            return max(1, int(self.base_line_width * avg_conf * 1.5))

        elif self.line_style == LineStyle.BODY_PART:
            # 身体部位分级
            if connection in self.HEAD_CONNECTIONS:
                return self.base_line_width + 1  # 中等
            elif connection in self.TORSO_CONNECTIONS:
                return self.base_line_width + 3  # 最粗
            elif connection in self.ARM_CONNECTIONS:
                return self.base_line_width      # 标准
            else:  # LEG_CONNECTIONS
                return self.base_line_width + 1  # 中等

        elif self.line_style == LineStyle.OPENPOSE:
            # OpenPose风格：躯干粗，四肢细，考虑置信度
            base = self.base_line_width

            if connection in self.TORSO_CONNECTIONS:
                base_width = base + 4  # 躯干最粗
            elif connection in self.HEAD_CONNECTIONS:
                base_width = base + 2  # 头部中等
            elif connection in self.LEG_CONNECTIONS:
                base_width = base + 1  # 腿部中等
            else:  # ARM_CONNECTIONS
                base_width = base      # 手臂较细

            # 应用置信度调制
            avg_conf = (conf1 + conf2) / 2
            return max(1, int(base_width * (0.5 + 0.5 * avg_conf)))

        elif self.line_style == LineStyle.THICK_TO_THIN:
            # 从躯干到四肢递减
            idx1, idx2 = connection

            # 计算关键点到躯干中心的平均距离
            # 躯干中心：肩膀和髋部的中点
            torso_keypoints = [5, 6, 11, 12]

            # 简化：根据关键点索引估算距离
            avg_idx = (idx1 + idx2) / 2

            if avg_idx < 7:  # 头部和肩膀
                return self.base_line_width + 3
            elif avg_idx < 13:  # 手臂和躯干
                return self.base_line_width + 2
            else:  # 腿部
                return self.base_line_width + 1

        else:
            return self.base_line_width

    def _draw_gradient_line(self,
                           image: np.ndarray,
                           pt1: Tuple[int, int],
                           pt2: Tuple[int, int],
                           color: Tuple[int, int, int],
                           max_width: int):
        """
        绘制渐变线条（OpenPose风格）
        从中点向两端渐细

        Args:
            image: 图像
            pt1, pt2: 两个端点
            color: 颜色
            max_width: 最大线宽
        """
        # 计算线段长度
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        length = np.sqrt(dx**2 + dy**2)

        if length < 1:
            return

        # 分段数量（根据长度）
        num_segments = max(5, int(length / 10))

        for i in range(num_segments):
            t1 = i / num_segments
            t2 = (i + 1) / num_segments

            # 计算当前段的两个端点
            x1 = int(pt1[0] + dx * t1)
            y1 = int(pt1[1] + dy * t1)
            x2 = int(pt1[0] + dx * t2)
            y2 = int(pt1[1] + dy * t2)

            # 计算当前段的线宽（从中点向两端渐细）
            # 使用高斯分布
            center_distance = abs(t1 + t2 - 1.0)  # 距离中点的距离（0-1）
            width_ratio = np.exp(-4 * center_distance**2)  # 高斯衰减
            width = max(1, int(max_width * width_ratio))

            cv2.line(image, (x1, y1), (x2, y2), color, width)

    def _draw_keypoints(self,
                       image: np.ndarray,
                       keypoints: np.ndarray,
                       confidences: np.ndarray,
                       colors: dict):
        """绘制关键点"""
        for i, (point, conf) in enumerate(zip(keypoints, confidences)):
            if conf > 0.5:
                x, y = int(point[0]), int(point[1])

                # 根据部位选择颜色
                if i < 5:  # 头部
                    color = colors.get('head', self.DEFAULT_COLORS['head'])
                    radius = 5
                elif i < 11:  # 手臂
                    color = colors.get('arms', self.DEFAULT_COLORS['arms'])
                    radius = 4
                elif i < 13:  # 躯干
                    color = colors.get('torso', self.DEFAULT_COLORS['torso'])
                    radius = 4
                else:  # 腿部
                    color = colors.get('legs', self.DEFAULT_COLORS['legs'])
                    radius = 4

                cv2.circle(image, (x, y), radius, color, -1)

                # 绘制外圈（增强可见性）
                cv2.circle(image, (x, y), radius + 1, (255, 255, 255), 1)

    def _draw_labels(self,
                    image: np.ndarray,
                    keypoints: np.ndarray,
                    confidences: np.ndarray,
                    names: List[str],
                    colors: dict):
        """绘制标签"""
        for i, (point, conf, name) in enumerate(zip(keypoints, confidences, names)):
            if conf > 0.5:
                x, y = int(point[0]), int(point[1])

                # 根据部位选择颜色
                if i < 5:
                    color = colors.get('head', self.DEFAULT_COLORS['head'])
                elif i < 11:
                    color = colors.get('arms', self.DEFAULT_COLORS['arms'])
                elif i < 13:
                    color = colors.get('torso', self.DEFAULT_COLORS['torso'])
                else:
                    color = colors.get('legs', self.DEFAULT_COLORS['legs'])

                # 绘制文字背景（提高可读性）
                text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(image,
                            (x + 10, y - 15),
                            (x + 10 + text_size[0], y - 15 + text_size[1]),
                            (0, 0, 0), -1)

                # 绘制文字
                cv2.putText(image, name, (x + 10, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


# 便捷函数
def draw_skeleton_with_style(image: np.ndarray,
                             keypoints: np.ndarray,
                             confidences: np.ndarray,
                             style: str = "openpose",
                             base_line_width: int = 2,
                             colors: dict = None,
                             show_keypoints: bool = True,
                             show_labels: bool = False) -> np.ndarray:
    """
    便捷函数：使用指定样式绘制骨架

    Args:
        image: 输入图像
        keypoints: 关键点 (N, 17, 2)
        confidences: 置信度 (N, 17)
        style: 样式名称 ("uniform", "confidence", "body_part", "openpose", "thick_to_thin")
        base_line_width: 基础线宽
        colors: 颜色字典
        show_keypoints: 显示关键点
        show_labels: 显示标签

    Returns:
        绘制后的图像
    """
    renderer = SkeletonRenderer(line_style=style, base_line_width=base_line_width)
    return renderer.draw_skeleton(image, keypoints, confidences, colors,
                                  show_keypoints, show_labels)


if __name__ == "__main__":
    """测试不同样式"""
    print("骨架渲染器模块")
    print("支持的样式：")
    for style in LineStyle:
        print(f"  - {style.value}")
