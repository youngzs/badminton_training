"""
羽毛球装备检测器（球拍和羽毛球）

使用YOLOv8物体检测模型检测羽毛球拍和羽毛球
"""

import torch
from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2
from PIL import Image, ImageDraw, ImageFont


class SportsEquipmentDetector:
    """羽毛球装备检测器"""

    def __init__(self,
                 model_path: str = "yolov8n.pt",
                 conf_threshold: float = 0.3,
                 use_gpu: bool = True):
        """
        初始化检测器

        Args:
            model_path: YOLOv8模型路径
            conf_threshold: 置信度阈值
            use_gpu: 是否使用GPU
        """
        self.conf_threshold = conf_threshold
        self.use_gpu = use_gpu and torch.cuda.is_available()

        # 加载模型
        self.model = YOLO(model_path)

        if self.use_gpu:
            self.model.to('cuda')
            print(f"✓ 装备检测器使用GPU")
        else:
            print(f"✓ 装备检测器使用CPU")

        # COCO数据集中的相关类别
        # 注意：标准COCO不包含羽毛球拍和羽毛球
        # 我们使用"sports ball"作为羽毛球的近似检测
        # 使用"tennis racket"作为球拍的近似检测
        self.class_mapping = {
            'sports ball': 'shuttlecock',  # 32: sports ball -> 羽毛球
            'tennis racket': 'racket',      # 38: tennis racket -> 球拍
        }

        # 获取模型的类别名称
        self.class_names = self.model.names

        # 找出我们关心的类别ID
        self.target_classes = []
        for class_id, class_name in self.class_names.items():
            if class_name in self.class_mapping:
                self.target_classes.append(class_id)

        print(f"✓ 目标类别: {[self.class_names[c] for c in self.target_classes]}")

    def detect(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        检测羽毛球装备

        Args:
            frame: 输入图像帧

        Returns:
            detections: 检测结果列表，每个元素包含：
                - class_name: 类别名称（'racket'或'shuttlecock'）
                - confidence: 置信度
                - bbox: 边界框 [x1, y1, x2, y2]
            annotated_frame: 标注后的图像
        """
        # 运行检测
        results = self.model(frame, conf=self.conf_threshold, verbose=False)

        detections = []
        annotated_frame = frame.copy()

        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    # 只处理我们关心的类别
                    if class_id in self.target_classes:
                        original_class = self.class_names[class_id]
                        mapped_class = self.class_mapping[original_class]

                        detections.append({
                            'class_name': mapped_class,
                            'original_class': original_class,
                            'confidence': float(conf),
                            'bbox': box.tolist()
                        })

        return detections, annotated_frame

    def draw_detections(self,
                       frame: np.ndarray,
                       detections: List[Dict],
                       draw_boxes: bool = True,
                       draw_labels: bool = True,
                       draw_confidence: bool = True,
                       box_color: Tuple[int, int, int] = (0, 255, 255),  # 黄色
                       text_color: Tuple[int, int, int] = (0, 0, 0),     # 黑色
                       line_width: int = 2) -> np.ndarray:
        """
        在图像上绘制检测结果

        Args:
            frame: 输入图像
            detections: 检测结果列表
            draw_boxes: 是否绘制边界框
            draw_labels: 是否绘制标签
            draw_confidence: 是否绘制置信度
            box_color: 边界框颜色 (B, G, R)
            text_color: 文字颜色 (B, G, R)
            line_width: 线条宽度

        Returns:
            annotated_frame: 标注后的图像
        """
        annotated_frame = frame.copy()

        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)

            # 绘制边界框
            if draw_boxes:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2),
                            box_color, line_width)

            # 准备标签文本
            if draw_labels or draw_confidence:
                label_parts = []

                if draw_labels:
                    # 中文标签
                    class_name_cn = {
                        'racket': '球拍',
                        'shuttlecock': '羽毛球'
                    }
                    label_parts.append(class_name_cn.get(det['class_name'], det['class_name']))

                if draw_confidence:
                    label_parts.append(f"{det['confidence']:.2f}")

                label = ' '.join(label_parts)

                # 使用PIL绘制中文文字
                annotated_frame = self._draw_chinese_text(
                    annotated_frame, label, (x1, y1 - 25),
                    font_size=20, text_color=text_color, bg_color=box_color
                )

        return annotated_frame

    def _draw_chinese_text(self,
                          img: np.ndarray,
                          text: str,
                          position: Tuple[int, int],
                          font_size: int = 20,
                          text_color: Tuple[int, int, int] = (0, 0, 0),
                          bg_color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        """
        使用PIL在图像上绘制中文文字

        Args:
            img: OpenCV图像（BGR格式）
            text: 要绘制的文字
            position: 文字位置(x, y)
            font_size: 字体大小
            text_color: 文字颜色(B, G, R)
            bg_color: 背景颜色(B, G, R)

        Returns:
            绘制后的图像
        """
        # 转换为PIL图像（RGB格式）
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 尝试加载中文字体
        try:
            # Windows系统字体路径
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
                "C:/Windows/Fonts/simhei.ttf",    # 黑体
                "C:/Windows/Fonts/simsun.ttc",    # 宋体
            ]
            font = None
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except:
                    continue

            if font is None:
                # 使用默认字体
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()

        # 获取文字边界框
        bbox = draw.textbbox(position, text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 绘制背景矩形（转换颜色为RGB）
        bg_color_rgb = (bg_color[2], bg_color[1], bg_color[0])
        padding = 4
        draw.rectangle(
            [position[0] - padding,
             position[1] - padding,
             position[0] + text_width + padding,
             position[1] + text_height + padding],
            fill=bg_color_rgb
        )

        # 绘制文字（转换颜色为RGB）
        text_color_rgb = (text_color[2], text_color[1], text_color[0])
        draw.text(position, text, font=font, fill=text_color_rgb)

        # 转换回OpenCV格式（BGR）
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


class CustomSportsDetector:
    """
    自定义羽毛球装备检测器

    如果您有专门训练的羽毛球拍和羽毛球检测模型，可以使用此类
    """

    def __init__(self, model_path: str, use_gpu: bool = True):
        """
        初始化自定义检测器

        Args:
            model_path: 自定义YOLOv8模型路径
            use_gpu: 是否使用GPU
        """
        self.model = YOLO(model_path)

        if use_gpu and torch.cuda.is_available():
            self.model.to('cuda')
            print(f"✓ 自定义检测器使用GPU")
        else:
            print(f"✓ 自定义检测器使用CPU")

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.3):
        """
        使用自定义模型检测

        Args:
            frame: 输入图像
            conf_threshold: 置信度阈值

        Returns:
            results: YOLOv8检测结果
        """
        return self.model(frame, conf=conf_threshold, verbose=False)


def create_detector(detector_type: str = "standard",
                   model_path: Optional[str] = None,
                   conf_threshold: float = 0.3,
                   use_gpu: bool = True) -> SportsEquipmentDetector:
    """
    创建检测器工厂函数

    Args:
        detector_type: 检测器类型 ("standard" 或 "custom")
        model_path: 模型路径（custom类型必须提供）
        conf_threshold: 置信度阈值
        use_gpu: 是否使用GPU

    Returns:
        detector: 检测器实例
    """
    if detector_type == "standard":
        # 使用标准YOLOv8模型（近似检测）
        if model_path is None:
            model_path = "yolov8n.pt"
        return SportsEquipmentDetector(model_path, conf_threshold, use_gpu)

    elif detector_type == "custom":
        # 使用自定义训练的模型
        if model_path is None:
            raise ValueError("自定义检测器必须提供model_path")
        return CustomSportsDetector(model_path, use_gpu)

    else:
        raise ValueError(f"未知的检测器类型: {detector_type}")


if __name__ == "__main__":
    # 测试代码
    import sys

    print("羽毛球装备检测器测试")
    print("=" * 50)

    # 创建检测器
    detector = create_detector(detector_type="standard", use_gpu=True)

    # 如果提供了测试图片
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        print(f"\n测试图片: {test_image}")

        frame = cv2.imread(test_image)
        if frame is None:
            print(f"错误：无法读取图片 {test_image}")
            sys.exit(1)

        # 检测
        detections, _ = detector.detect(frame)

        print(f"\n检测结果:")
        print(f"  检测到 {len(detections)} 个物体")

        for i, det in enumerate(detections):
            print(f"\n  物体 {i+1}:")
            print(f"    类别: {det['class_name']}")
            print(f"    原始类别: {det['original_class']}")
            print(f"    置信度: {det['confidence']:.3f}")
            print(f"    边界框: {det['bbox']}")

        # 绘制结果
        annotated = detector.draw_detections(frame, detections)

        output_path = test_image.replace('.', '_detected.')
        cv2.imwrite(output_path, annotated)
        print(f"\n结果已保存到: {output_path}")
    else:
        print("\n使用方法:")
        print("  python sports_equipment_detector.py <test_image.jpg>")
        print("\n注意:")
        print("  - 标准模型使用COCO数据集，可以检测'sports ball'和'tennis racket'")
        print("  - 如需更准确的羽毛球拍和羽毛球检测，建议训练自定义模型")
