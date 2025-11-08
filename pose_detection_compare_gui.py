# -*- coding: utf-8 -*-
"""
羽毛球训练视频对比分析系统 - 增强GUI
支持原视频与分析结果对比、参数调整、配置保存
"""

import sys
import io
import os

# 设置Windows控制台输出编码
if sys.platform == 'win32':
    try:
        if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageTk
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List

# 抑制OpenCV的H.264解码警告（不影响功能的技术性警告）
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'  # 只显示严重错误
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from pose_detection_training import GPUOptimizedPoseDetector, PoseAnalyzer
except ImportError:
    print("警告: 无法导入训练模块，部分功能可能不可用")

try:
    from sports_equipment_detector import SportsEquipmentDetector
except ImportError:
    print("警告: 无法导入装备检测模块，装备检测功能将不可用")
    SportsEquipmentDetector = None

try:
    from mediapipe_detector import MediaPipePoseDetector
except ImportError:
    print("警告: 无法导入MediaPipe检测模块")
    MediaPipePoseDetector = None


@dataclass
class ProcessConfig:
    """处理配置"""
    # 模型设置
    pose_engine: str = "yolov8"  # 姿态检测引擎: "yolov8" 或 "mediapipe"
    pose_model: str = "yolov8n-pose.pt"  # YOLOv8模型路径
    mediapipe_complexity: int = 1  # MediaPipe模型复杂度 (0=Lite, 1=Full, 2=Heavy)
    skeleton_style: str = "openpose"  # 骨架线条样式: "uniform", "confidence", "body_part", "openpose", "thick_to_thin"
    confidence: float = 0.25
    use_gpu: bool = True
    use_fp16: bool = True

    # 显示设置
    line_width: int = 2
    keypoint_size: int = 4
    show_labels: bool = False
    show_boxes: bool = False          # 显示人物边界框
    show_confidence: bool = False     # 显示置信度分数
    show_skeleton: bool = True        # 显示骨架线条
    show_keypoints: bool = True       # 显示关键点

    # 装备检测设置
    enable_equipment_detection: bool = True   # 启用装备检测
    show_equipment_boxes: bool = True         # 显示装备边界框
    show_equipment_labels: bool = True        # 显示装备标签
    equipment_confidence: float = 0.3         # 装备检测置信度

    # 颜色设置（RGBA格式）
    color_head: tuple = (255, 0, 0, 255)      # 红色 - 头部
    color_arms: tuple = (0, 255, 0, 255)      # 绿色 - 手臂
    color_torso: tuple = (0, 0, 255, 255)     # 蓝色 - 躯干
    color_legs: tuple = (255, 255, 0, 255)    # 黄色 - 腿部

    # 分析设置
    enable_analysis: bool = True
    show_angles: bool = True
    show_issues: bool = True

    # 慢动作设置
    slowmo_factor: float = 1.0

    # 输出设置
    save_keyframes: bool = False
    video_quality: int = 90  # 0-100

    # 历史记录（最多保存20条）
    processing_history: list = None

    def __post_init__(self):
        """初始化后处理"""
        if self.processing_history is None:
            self.processing_history = []

    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict):
        """从字典创建"""
        return cls(**data)


class VideoComparePlayer:
    """视频对比播放器"""

    def __init__(self, canvas_original: tk.Canvas, canvas_processed: tk.Canvas,
                 canvas_pose: tk.Canvas, frame_label: tk.Label = None):
        self.canvas_original = canvas_original
        self.canvas_processed = canvas_processed
        self.canvas_pose = canvas_pose
        self.frame_label = frame_label

        self.video_original = None
        self.video_processed = None

        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30

        self.play_thread = None

        # 姿态数据
        self.pose_data_dict = {}  # {frame_number: pose_info}

        # 进度更新回调函数
        self.update_callback = None

    def load_original_video(self, video_path: str):
        """加载原始视频"""
        if self.video_original:
            self.video_original.release()

        self.video_original = cv2.VideoCapture(video_path)
        self.fps = int(self.video_original.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.video_original.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0

        return self.total_frames, self.fps

    def load_processed_video(self, video_path: str):
        """加载处理后的视频"""
        if self.video_processed:
            self.video_processed.release()

        self.video_processed = cv2.VideoCapture(video_path)

    def load_pose_data(self, pose_data_file: str):
        """加载姿态分析数据"""
        try:
            import json
            with open(pose_data_file, 'r', encoding='utf-8') as f:
                pose_data_list = json.load(f)

            # 转换为字典，方便根据帧号查询
            self.pose_data_dict = {}
            for item in pose_data_list:
                frame_num = item.get('frame', 0)
                pose_info = item.get('pose_info', {})
                self.pose_data_dict[frame_num] = pose_info

            print(f"✓ 加载了 {len(self.pose_data_dict)} 帧的姿态数据")
        except Exception as e:
            print(f"⚠ 加载姿态数据失败: {e}")
            self.pose_data_dict = {}

    def seek(self, frame_number: int):
        """跳转到指定帧（优化的H.264兼容方法）"""
        self.current_frame = max(0, min(frame_number, self.total_frames - 1))

        # 改进的跳转策略：
        # 1. 对于向后小幅度跳转（<10帧），逐帧读取更准确
        # 2. 对于大幅度跳转，使用CAP_PROP_POS_FRAMES但接受可能的警告
        # 3. H.264警告是技术性的，不影响实际显示

        if self.video_original:
            current_pos = int(self.video_original.get(cv2.CAP_PROP_POS_FRAMES))
            diff = self.current_frame - current_pos

            # 小幅度向前跳转：逐帧读取
            if 0 < diff < 10:
                for _ in range(diff):
                    self.video_original.read()
            else:
                # 大幅度跳转：直接设置位置
                self.video_original.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

        if self.video_processed:
            current_pos = int(self.video_processed.get(cv2.CAP_PROP_POS_FRAMES))
            diff = self.current_frame - current_pos

            if 0 < diff < 10:
                for _ in range(diff):
                    self.video_processed.read()
            else:
                self.video_processed.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

    def get_current_frames(self):
        """获取当前帧"""
        frame_orig = None
        frame_proc = None

        if self.video_original:
            ret, frame_orig = self.video_original.read()
            if not ret:
                frame_orig = None

        if self.video_processed:
            ret, frame_proc = self.video_processed.read()
            if not ret:
                frame_proc = None

        return frame_orig, frame_proc

    def display_frames(self, frame_orig, frame_proc, pose_data=None):
        """显示帧到画布"""
        # 显示原始视频
        if frame_orig is not None:
            self._display_frame_on_canvas(frame_orig, self.canvas_original)

        # 显示处理后的视频
        if frame_proc is not None:
            self._display_frame_on_canvas(frame_proc, self.canvas_processed)

        # 显示姿态信息
        if pose_data is not None:
            self._display_pose_data(pose_data, self.canvas_pose)

    def _display_frame_on_canvas(self, frame, canvas):
        """在画布上显示帧"""
        # 转换颜色空间
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 获取画布尺寸
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 640
            canvas_height = 480

        # 调整尺寸
        h, w = frame_rgb.shape[:2]
        scale = min(canvas_width / w, canvas_height / h)
        new_w, new_h = int(w * scale), int(h * scale)

        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))

        # 转换为PIL Image
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)

        # 显示
        canvas.delete("all")
        x = (canvas_width - new_w) // 2
        y = (canvas_height - new_h) // 2
        canvas.create_image(x, y, anchor=tk.NW, image=imgtk)
        canvas.image = imgtk  # 保持引用

    def _display_pose_data(self, pose_data, canvas):
        """显示姿态数据"""
        canvas.delete("all")

        if not pose_data or not pose_data.get("valid"):
            canvas.create_text(320, 240, text="无姿态数据",
                             font=("Arial", 16), fill="#666")
            return

        y_offset = 20

        # 标题
        canvas.create_text(20, y_offset, anchor="nw", text="姿态分析",
                          font=("Arial", 18, "bold"), fill="#2c3e50")
        y_offset += 40

        # 显示角度
        if "angles" in pose_data and pose_data["angles"]:
            canvas.create_text(20, y_offset, anchor="nw", text="关键角度:",
                             font=("Arial", 14, "bold"), fill="#34495e")
            y_offset += 30

            for joint, angle in pose_data["angles"].items():
                # 判断角度是否正常
                color = "#27ae60" if self._is_angle_normal(joint, angle) else "#e74c3c"
                status = "✓" if color == "#27ae60" else "✗"

                text = f"{status} {self._translate_joint(joint)}: {angle:.1f}°"
                canvas.create_text(40, y_offset, anchor="nw", text=text,
                                 font=("Arial", 12), fill=color)
                y_offset += 25

        y_offset += 10

        # 显示问题
        if "issues" in pose_data and pose_data["issues"]:
            canvas.create_text(20, y_offset, anchor="nw", text="检测到的问题:",
                             font=("Arial", 14, "bold"), fill="#e74c3c")
            y_offset += 30

            for i, issue in enumerate(pose_data["issues"][:3]):  # 最多显示3个
                canvas.create_text(40, y_offset, anchor="nw",
                                 text=f"• {issue}",
                                 font=("Arial", 11), fill="#c0392b")
                y_offset += 25

        y_offset += 10

        # 显示建议
        if "suggestions" in pose_data and pose_data["suggestions"]:
            canvas.create_text(20, y_offset, anchor="nw", text="改进建议:",
                             font=("Arial", 14, "bold"), fill="#3498db")
            y_offset += 30

            for i, suggestion in enumerate(pose_data["suggestions"][:2]):  # 最多显示2个
                canvas.create_text(40, y_offset, anchor="nw",
                                 text=f"• {suggestion}",
                                 font=("Arial", 11), fill="#2980b9",
                                 width=580)  # 自动换行
                y_offset += 40

    def _is_angle_normal(self, joint: str, angle: float) -> bool:
        """判断角度是否正常"""
        normal_ranges = {
            "right_elbow": (140, 180),
            "left_elbow": (140, 180),
            "right_knee": (145, 165),
            "left_knee": (145, 165),
            "torso_lean": (0, 20),
        }

        if joint not in normal_ranges:
            return True

        min_angle, max_angle = normal_ranges[joint]
        return min_angle <= angle <= max_angle

    def _translate_joint(self, joint: str) -> str:
        """翻译关节名称"""
        translations = {
            "right_elbow": "右肘",
            "left_elbow": "左肘",
            "right_knee": "右膝",
            "left_knee": "左膝",
            "torso_lean": "躯干倾斜",
        }
        return translations.get(joint, joint)

    def play(self):
        """播放"""
        # 防止重复创建线程
        if not self.is_playing:
            # 如果有正在运行的线程，等待其结束
            if self.play_thread and self.play_thread.is_alive():
                return

            self.is_playing = True
            self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
            self.play_thread.start()

    def pause(self):
        """暂停"""
        self.is_playing = False
        # 等待播放线程结束
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=0.5)

    def _play_loop(self):
        """播放循环"""
        while self.is_playing and self.current_frame < self.total_frames:
            frame_orig, frame_proc = self.get_current_frames()

            # 获取当前帧的姿态数据
            pose_data = self.pose_data_dict.get(self.current_frame + 1, None)  # 帧号从1开始

            if frame_orig is not None or frame_proc is not None:
                self.display_frames(frame_orig, frame_proc, pose_data)

            # 更新进度条和帧标签（通过回调函数）
            if self.update_callback:
                self.update_callback(self.current_frame, self.total_frames)

            self.current_frame += 1
            time.sleep(1.0 / self.fps)

        self.is_playing = False

    def release(self):
        """释放资源"""
        self.is_playing = False
        if self.video_original:
            self.video_original.release()
        if self.video_processed:
            self.video_processed.release()


class CompareGUI:
    """视频对比分析GUI"""

    def __init__(self, root):
        self.root = root
        self.root.title("羽毛球训练视频对比分析系统")
        self.root.geometry("1600x900")

        # 配置
        self.config = ProcessConfig()
        self.config_file = Path("compare_config.json")

        # 视频播放器
        self.player = None

        # 处理状态
        self.is_processing = False
        self.process_thread = None

        # 播放控制防抖动
        self.last_toggle_time = 0
        self.toggle_debounce_delay = 0.3  # 300ms防抖动延迟

        # 创建界面
        self.create_widgets()

        # 加载配置
        self.load_config()

        # 刷新历史记录
        self.refresh_history()

        # 绑定键盘快捷键（空格键播放/暂停）
        self.root.bind('<space>', lambda e: self.toggle_play_pause())

        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        """创建界面组件"""
        # 主容器
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左侧：视频对比区域
        left_frame = ttk.Frame(main_container)
        main_container.add(left_frame, weight=3)
        self.create_video_area(left_frame)

        # 右侧：参数设置区域
        right_frame = ttk.Frame(main_container)
        main_container.add(right_frame, weight=1)
        self.create_settings_area(right_frame)

    def create_video_area(self, parent):
        """创建视频区域"""
        # 顶部：文件选择
        file_frame = ttk.LabelFrame(parent, text="文件选择", padding=10)
        file_frame.pack(fill=tk.X, padx=5, pady=5)

        # 输入视频
        ttk.Label(file_frame, text="输入视频:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.input_video_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.input_video_var, width=60).grid(
            row=0, column=1, padx=5)
        ttk.Button(file_frame, text="浏览",
                  command=self.browse_input_video).grid(row=0, column=2, padx=5)

        # 输出视频（可选）
        ttk.Label(file_frame, text="输出视频:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.output_video_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.output_video_var, width=60).grid(
            row=1, column=1, padx=5)
        ttk.Button(file_frame, text="浏览",
                  command=self.browse_output_video).grid(row=1, column=2, padx=5)

        # 中部：视频显示区域
        video_container = ttk.Frame(parent)
        video_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建三个显示区域
        # 左：原始视频
        orig_frame = ttk.LabelFrame(video_container, text="原始视频", padding=5)
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.canvas_original = tk.Canvas(orig_frame, bg="#2c3e50", highlightthickness=0)
        self.canvas_original.pack(fill=tk.BOTH, expand=True)

        # 中：处理后的视频
        proc_frame = ttk.LabelFrame(video_container, text="分析结果", padding=5)
        proc_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.canvas_processed = tk.Canvas(proc_frame, bg="#2c3e50", highlightthickness=0)
        self.canvas_processed.pack(fill=tk.BOTH, expand=True)

        # 右：姿态信息
        pose_frame = ttk.LabelFrame(video_container, text="姿态分析", padding=5)
        pose_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.canvas_pose = tk.Canvas(pose_frame, bg="#ecf0f1", highlightthickness=0)
        self.canvas_pose.pack(fill=tk.BOTH, expand=True)

        # 底部：播放控制
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # 播放控制按钮
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(side=tk.LEFT)

        # 播放/暂停切换按钮
        self.play_pause_btn = ttk.Button(btn_frame, text="▶ 播放",
                                         command=self.toggle_play_pause)
        self.play_pause_btn.pack(side=tk.LEFT, padx=2)

        ttk.Button(btn_frame, text="⏮ 重置",
                  command=self.reset_video).pack(side=tk.LEFT, padx=2)

        # 进度条
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_scale = ttk.Scale(control_frame, from_=0, to=100,
                                       variable=self.progress_var,
                                       orient=tk.HORIZONTAL,
                                       command=self.on_progress_change)
        self.progress_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        # 帧信息
        self.frame_label = ttk.Label(control_frame, text="0 / 0")
        self.frame_label.pack(side=tk.LEFT, padx=5)

        # 初始化播放器（在frame_label创建之后）
        self.player = VideoComparePlayer(
            self.canvas_original,
            self.canvas_processed,
            self.canvas_pose,
            self.frame_label
        )

        # 设置进度更新回调
        self.player.update_callback = self.update_playback_progress

    def create_settings_area(self, parent):
        """创建设置区域"""
        # Notebook (标签页)
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 标签页1：模型设置
        model_frame = ttk.Frame(notebook)
        notebook.add(model_frame, text="模型设置")
        self.create_model_settings(model_frame)

        # 标签页2：显示设置
        display_frame = ttk.Frame(notebook)
        notebook.add(display_frame, text="显示设置")
        self.create_display_settings(display_frame)

        # 标签页3：颜色设置
        color_frame = ttk.Frame(notebook)
        notebook.add(color_frame, text="颜色设置")
        self.create_color_settings(color_frame)

        # 标签页4：分析设置
        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="分析设置")
        self.create_analysis_settings(analysis_frame)

        # 底部：操作按钮（紧凑布局）
        action_frame = ttk.LabelFrame(parent, text="操作", padding=5)
        action_frame.pack(fill=tk.X, padx=5, pady=3)

        # 第一行：处理控制
        row1 = ttk.Frame(action_frame)
        row1.pack(fill=tk.X, pady=1)
        ttk.Button(row1, text="开始处理",
                  command=self.start_processing).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(row1, text="停止处理",
                  command=self.stop_processing).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        # 第二行：历史记录下拉选择
        row2 = ttk.Frame(action_frame)
        row2.pack(fill=tk.X, pady=1)
        ttk.Label(row2, text="历史:", width=5).pack(side=tk.LEFT)
        self.history_var = tk.StringVar()
        self.history_combo = ttk.Combobox(row2, textvariable=self.history_var,
                                         state='readonly', width=30)
        self.history_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        self.history_combo.bind('<<ComboboxSelected>>', self.load_from_history)
        ttk.Button(row2, text="刷新", width=5,
                  command=self.refresh_history).pack(side=tk.LEFT, padx=1)

        # 第三行：配置管理
        row3 = ttk.Frame(action_frame)
        row3.pack(fill=tk.X, pady=1)
        ttk.Button(row3, text="保存配置", width=10,
                  command=self.save_config).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(row3, text="加载配置", width=10,
                  command=self.load_config).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(row3, text="重置配置", width=10,
                  command=self.reset_config).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        # 进度信息
        progress_info_frame = ttk.LabelFrame(parent, text="处理进度", padding=10)
        progress_info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.status_text = tk.Text(progress_info_frame, height=10, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(progress_info_frame, command=self.status_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.config(yscrollcommand=scrollbar.set)

    def create_model_settings(self, parent):
        """创建模型设置"""
        frame = ttk.Frame(parent, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        row = 0

        # 检测引擎选择
        ttk.Label(frame, text="检测引擎:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.engine_var = tk.StringVar(value=self.config.pose_engine)
        engine_combo = ttk.Combobox(frame, textvariable=self.engine_var,
                                    values=["yolov8", "mediapipe"],
                                    state="readonly", width=20)
        engine_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        engine_combo.bind('<<ComboboxSelected>>', self.on_engine_changed)

        # 添加说明标签
        info_text = "YOLOv8: 多人/快速 | MediaPipe: 单人/高精度"
        ttk.Label(frame, text=info_text, foreground="gray", font=("", 8)).grid(
            row=row, column=2, sticky=tk.W, padx=5)
        row += 1

        # YOLOv8模型选择（仅在YOLOv8引擎时显示）
        ttk.Label(frame, text="YOLOv8模型:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar(value=self.config.pose_model)
        self.model_combo = ttk.Combobox(frame, textvariable=self.model_var,
                                   values=["yolov8n-pose.pt", "yolov8s-pose.pt",
                                          "yolov8m-pose.pt", "yolov8l-pose.pt"],
                                   state="readonly", width=20)
        self.model_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        self.model_label = ttk.Label(frame, text="YOLOv8模型:")
        self.model_label.grid(row=row, column=0, sticky=tk.W, pady=5)
        row += 1

        # MediaPipe模型复杂度（仅在MediaPipe引擎时显示）
        self.mediapipe_label = ttk.Label(frame, text="MediaPipe复杂度:")
        self.mediapipe_label.grid(row=row, column=0, sticky=tk.W, pady=5)
        self.mediapipe_var = tk.IntVar(value=self.config.mediapipe_complexity)
        self.mediapipe_combo = ttk.Combobox(frame, textvariable=self.mediapipe_var,
                                           values=[0, 1, 2],
                                           state="readonly", width=20)
        self.mediapipe_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)

        # 添加说明
        mediapipe_info = "0=快速 | 1=平衡 | 2=精确"
        self.mediapipe_info_label = ttk.Label(frame, text=mediapipe_info,
                                             foreground="gray", font=("", 8))
        self.mediapipe_info_label.grid(row=row, column=2, sticky=tk.W, padx=5)
        row += 1

        # 初始化控件可见性
        self.update_model_controls()

        # 骨架线条样式
        ttk.Label(frame, text="线条样式:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.skeleton_style_var = tk.StringVar(value=self.config.skeleton_style)
        style_combo = ttk.Combobox(frame, textvariable=self.skeleton_style_var,
                                   values=["uniform", "confidence", "body_part", "openpose", "thick_to_thin"],
                                   state="readonly", width=20)
        style_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)

        # 样式说明
        style_info = "OpenPose风格 | 推荐"
        ttk.Label(frame, text=style_info, foreground="gray", font=("", 8)).grid(
            row=row, column=2, sticky=tk.W, padx=5)
        row += 1

        # 置信度
        ttk.Label(frame, text="检测置信度:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.confidence_var = tk.DoubleVar(value=self.config.confidence)
        confidence_scale = ttk.Scale(frame, from_=0.1, to=0.9,
                                     variable=self.confidence_var,
                                     orient=tk.HORIZONTAL)
        confidence_scale.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        self.confidence_label = ttk.Label(frame, text=f"{self.config.confidence:.2f}")
        self.confidence_label.grid(row=row, column=2, padx=5)
        confidence_scale.configure(command=lambda v: self.confidence_label.config(
            text=f"{float(v):.2f}"))
        row += 1

        # GPU设置
        ttk.Label(frame, text="GPU加速:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.gpu_var = tk.BooleanVar(value=self.config.use_gpu)
        ttk.Checkbutton(frame, variable=self.gpu_var).grid(row=row, column=1, sticky=tk.W)
        row += 1

        # FP16设置
        ttk.Label(frame, text="FP16加速:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.fp16_var = tk.BooleanVar(value=self.config.use_fp16)
        ttk.Checkbutton(frame, variable=self.fp16_var).grid(row=row, column=1, sticky=tk.W)
        row += 1

        frame.columnconfigure(1, weight=1)

    def create_display_settings(self, parent):
        """创建显示设置"""
        frame = ttk.Frame(parent, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        row = 0

        # 线条宽度
        ttk.Label(frame, text="线条宽度:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.line_width_var = tk.IntVar(value=self.config.line_width)
        line_scale = ttk.Scale(frame, from_=1, to=10, variable=self.line_width_var,
                              orient=tk.HORIZONTAL)
        line_scale.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        self.line_width_label = ttk.Label(frame, text=str(self.config.line_width))
        self.line_width_label.grid(row=row, column=2, padx=5)
        line_scale.configure(command=lambda v: self.line_width_label.config(
            text=str(int(float(v)))))
        row += 1

        # 关键点大小
        ttk.Label(frame, text="关键点大小:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.keypoint_size_var = tk.IntVar(value=self.config.keypoint_size)
        kp_scale = ttk.Scale(frame, from_=2, to=15, variable=self.keypoint_size_var,
                            orient=tk.HORIZONTAL)
        kp_scale.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        self.kp_size_label = ttk.Label(frame, text=str(self.config.keypoint_size))
        self.kp_size_label.grid(row=row, column=2, padx=5)
        kp_scale.configure(command=lambda v: self.kp_size_label.config(
            text=str(int(float(v)))))
        row += 1

        # 显示标签
        ttk.Label(frame, text="显示标签:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.show_labels_var = tk.BooleanVar(value=self.config.show_labels)
        ttk.Checkbutton(frame, variable=self.show_labels_var).grid(
            row=row, column=1, sticky=tk.W)
        row += 1

        # 显示边界框
        ttk.Label(frame, text="显示边界框:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.show_boxes_var = tk.BooleanVar(value=self.config.show_boxes)
        ttk.Checkbutton(frame, variable=self.show_boxes_var).grid(
            row=row, column=1, sticky=tk.W)
        row += 1

        # 显示置信度
        ttk.Label(frame, text="显示置信度:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.show_confidence_var = tk.BooleanVar(value=self.config.show_confidence)
        ttk.Checkbutton(frame, variable=self.show_confidence_var).grid(
            row=row, column=1, sticky=tk.W)
        row += 1

        # 显示骨架线条
        ttk.Label(frame, text="显示骨架线条:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.show_skeleton_var = tk.BooleanVar(value=self.config.show_skeleton)
        ttk.Checkbutton(frame, variable=self.show_skeleton_var).grid(
            row=row, column=1, sticky=tk.W)
        row += 1

        # 显示关键点
        ttk.Label(frame, text="显示关键点:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.show_keypoints_var = tk.BooleanVar(value=self.config.show_keypoints)
        ttk.Checkbutton(frame, variable=self.show_keypoints_var).grid(
            row=row, column=1, sticky=tk.W)
        row += 1

        # 分隔线
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2,
                                                         sticky=(tk.W, tk.E), pady=10)
        row += 1

        # 装备检测标题
        ttk.Label(frame, text="装备检测设置:", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1

        # 启用装备检测
        ttk.Label(frame, text="检测球拍和球:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.equipment_detect_var = tk.BooleanVar(value=self.config.enable_equipment_detection)
        ttk.Checkbutton(frame, variable=self.equipment_detect_var).grid(
            row=row, column=1, sticky=tk.W)
        row += 1

        # 显示装备边界框
        ttk.Label(frame, text="显示装备框:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.equipment_boxes_var = tk.BooleanVar(value=self.config.show_equipment_boxes)
        ttk.Checkbutton(frame, variable=self.equipment_boxes_var).grid(
            row=row, column=1, sticky=tk.W)
        row += 1

        # 显示装备标签
        ttk.Label(frame, text="显示装备标签:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.equipment_labels_var = tk.BooleanVar(value=self.config.show_equipment_labels)
        ttk.Checkbutton(frame, variable=self.equipment_labels_var).grid(
            row=row, column=1, sticky=tk.W)
        row += 1

        # 装备检测置信度
        ttk.Label(frame, text="装备检测阈值:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.equipment_conf_var = tk.DoubleVar(value=self.config.equipment_confidence)
        equipment_conf_scale = ttk.Scale(frame, from_=0.1, to=0.9,
                                        variable=self.equipment_conf_var,
                                        orient=tk.HORIZONTAL)
        equipment_conf_scale.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        row += 1

        # 装备置信度值显示
        equipment_conf_label = ttk.Label(frame, text="0.00")
        equipment_conf_label.grid(row=row, column=1, sticky=tk.W)

        def update_equipment_conf_label(*args):
            equipment_conf_label.config(text=f"{self.equipment_conf_var.get():.2f}")

        self.equipment_conf_var.trace_add('write', update_equipment_conf_label)
        update_equipment_conf_label()
        row += 1

        # 慢动作因子
        ttk.Label(frame, text="慢动作倍数:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.slowmo_var = tk.DoubleVar(value=self.config.slowmo_factor)
        slowmo_combo = ttk.Combobox(frame, textvariable=self.slowmo_var,
                                    values=[1.0, 0.5, 0.25, 0.125],
                                    state="readonly", width=10)
        slowmo_combo.grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        # 视频质量
        ttk.Label(frame, text="输出质量:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.quality_var = tk.IntVar(value=self.config.video_quality)
        quality_scale = ttk.Scale(frame, from_=50, to=100, variable=self.quality_var,
                                 orient=tk.HORIZONTAL)
        quality_scale.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        self.quality_label = ttk.Label(frame, text=str(self.config.video_quality))
        self.quality_label.grid(row=row, column=2, padx=5)
        quality_scale.configure(command=lambda v: self.quality_label.config(
            text=str(int(float(v)))))
        row += 1

        frame.columnconfigure(1, weight=1)

    def create_color_settings(self, parent):
        """创建颜色设置"""
        frame = ttk.Frame(parent, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        colors = [
            ("头部颜色:", "color_head", (255, 0, 0, 255)),
            ("手臂颜色:", "color_arms", (0, 255, 0, 255)),
            ("躯干颜色:", "color_torso", (0, 0, 255, 255)),
            ("腿部颜色:", "color_legs", (255, 255, 0, 255)),
        ]

        self.color_vars = {}

        for i, (label, key, default) in enumerate(colors):
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=5)

            # 颜色预览
            color_hex = f"#{default[0]:02x}{default[1]:02x}{default[2]:02x}"
            preview = tk.Canvas(frame, width=60, height=25, bg=color_hex,
                              highlightthickness=1, highlightbackground="#bdc3c7")
            preview.grid(row=i, column=1, padx=5, pady=5)

            # 选择按钮
            self.color_vars[key] = default
            ttk.Button(frame, text="选择颜色",
                      command=lambda k=key, p=preview: self.choose_color(k, p)).grid(
                row=i, column=2, padx=5)

    def create_analysis_settings(self, parent):
        """创建分析设置"""
        frame = ttk.Frame(parent, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        row = 0

        # 启用分析
        ttk.Label(frame, text="启用AI分析:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.analysis_var = tk.BooleanVar(value=self.config.enable_analysis)
        ttk.Checkbutton(frame, variable=self.analysis_var).grid(
            row=row, column=1, sticky=tk.W)
        row += 1

        # 显示角度
        ttk.Label(frame, text="显示角度:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.show_angles_var = tk.BooleanVar(value=self.config.show_angles)
        ttk.Checkbutton(frame, variable=self.show_angles_var).grid(
            row=row, column=1, sticky=tk.W)
        row += 1

        # 显示问题
        ttk.Label(frame, text="显示问题:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.show_issues_var = tk.BooleanVar(value=self.config.show_issues)
        ttk.Checkbutton(frame, variable=self.show_issues_var).grid(
            row=row, column=1, sticky=tk.W)
        row += 1

        # 保存关键帧
        ttk.Label(frame, text="保存关键帧:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.save_frames_var = tk.BooleanVar(value=self.config.save_keyframes)
        ttk.Checkbutton(frame, variable=self.save_frames_var).grid(
            row=row, column=1, sticky=tk.W)
        row += 1

    def choose_color(self, key, preview_canvas):
        """选择颜色"""
        from tkinter import colorchooser

        current_color = self.color_vars[key]
        color_hex = f"#{current_color[0]:02x}{current_color[1]:02x}{current_color[2]:02x}"

        color = colorchooser.askcolor(initialcolor=color_hex)
        if color[0]:
            r, g, b = [int(c) for c in color[0]]
            self.color_vars[key] = (r, g, b, 255)
            preview_canvas.config(bg=color[1])

    def browse_input_video(self):
        """浏览输入视频"""
        filename = filedialog.askopenfilename(
            title="选择输入视频",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")]
        )
        if filename:
            self.input_video_var.set(filename)

            # 自动设置输出路径
            if not self.output_video_var.get():
                input_path = Path(filename)
                self.output_video_var.set(
                    str(input_path.parent / f"{input_path.stem}_analyzed.mp4"))

            # 加载视频到播放器
            try:
                total_frames, fps = self.player.load_original_video(filename)
                self.frame_label.config(text=f"0 / {total_frames}")
                self.log_status(f"✓ 加载视频: {total_frames} 帧, {fps} FPS")
            except Exception as e:
                self.log_status(f"✗ 加载视频失败: {e}")

    def browse_output_video(self):
        """浏览输出视频"""
        filename = filedialog.asksaveasfilename(
            title="选择输出视频",
            defaultextension=".mp4",
            filetypes=[("MP4文件", "*.mp4"), ("所有文件", "*.*")]
        )
        if filename:
            self.output_video_var.set(filename)

    def toggle_play_pause(self):
        """切换播放/暂停状态"""
        import time

        # 防抖动：如果距离上次切换时间太短，忽略本次操作
        current_time = time.time()
        if current_time - self.last_toggle_time < self.toggle_debounce_delay:
            return

        self.last_toggle_time = current_time

        # 检查播放器是否存在
        if not self.player:
            return

        if self.player.is_playing:
            # 当前正在播放，切换为暂停
            self.player.pause()
            self.play_pause_btn.config(text="▶ 播放")
        else:
            # 当前暂停，切换为播放
            self.player.play()
            self.play_pause_btn.config(text="⏸ 暂停")

    def play_video(self):
        """播放视频"""
        self.player.play()
        self.play_pause_btn.config(text="⏸ 暂停")

    def pause_video(self):
        """暂停视频"""
        self.player.pause()
        self.play_pause_btn.config(text="▶ 播放")

    def reset_video(self):
        """重置视频"""
        self.player.pause()
        self.play_pause_btn.config(text="▶ 播放")
        self.player.seek(0)

        # 重置进度条
        self.progress_var.set(0)

        # 更新帧标签
        self.frame_label.config(text=f"1 / {self.player.total_frames}")

        # 显示第一帧
        frame_orig, frame_proc = self.player.get_current_frames()
        pose_data = self.player.pose_data_dict.get(1, None)  # 第1帧
        self.player.display_frames(frame_orig, frame_proc, pose_data)

    def on_progress_change(self, value):
        """进度条变化"""
        if self.player.total_frames > 0:
            frame_num = int(float(value) * self.player.total_frames / 100)
            self.player.seek(frame_num)
            frame_orig, frame_proc = self.player.get_current_frames()
            pose_data = self.player.pose_data_dict.get(frame_num + 1, None)
            self.player.display_frames(frame_orig, frame_proc, pose_data)
            self.frame_label.config(text=f"{frame_num + 1} / {self.player.total_frames}")

    def update_playback_progress(self, current_frame, total_frames):
        """播放时更新进度条和帧标签"""
        # 计算进度百分比
        if total_frames > 0:
            progress = (current_frame / total_frames) * 100
            self.progress_var.set(progress)
            # 更新帧标签（帧号从1开始显示）
            self.frame_label.config(text=f"{current_frame + 1} / {total_frames}")

    def start_processing(self):
        """开始处理"""
        if self.is_processing:
            return

        input_video = self.input_video_var.get()
        if not input_video or not Path(input_video).exists():
            messagebox.showerror("错误", "请选择有效的输入视频")
            return

        output_video = self.output_video_var.get()
        if not output_video:
            messagebox.showerror("错误", "请指定输出视频路径")
            return

        # 更新配置
        self.update_config_from_ui()

        # 启动处理线程
        self.is_processing = True
        self.process_thread = threading.Thread(target=self.process_video, daemon=True)
        self.process_thread.start()

    def stop_processing(self):
        """停止处理"""
        self.is_processing = False
        self.log_status("⏹ 用户中断处理")

    def process_video(self):
        """处理视频（后台线程）"""
        try:
            input_video = self.input_video_var.get()
            output_video = self.output_video_var.get()

            self.log_status(f"▶ 开始处理: {Path(input_video).name}")
            self.log_status(f"  输出: {Path(output_video).name}")

            # 初始化检测器和分析器
            self.log_status("⚙ 初始化姿态检测器...")

            # 根据配置选择检测引擎
            if self.config.pose_engine == "mediapipe":
                self.log_status(f"  引擎: MediaPipe Pose (复杂度={self.config.mediapipe_complexity})")
                self.log_status(f"  线条样式: {self.config.skeleton_style}")
                detector = MediaPipePoseDetector(
                    min_detection_confidence=self.config.confidence,
                    min_tracking_confidence=self.config.confidence,
                    model_complexity=self.config.mediapipe_complexity,
                    line_style=self.config.skeleton_style
                )
            else:  # yolov8
                self.log_status(f"  引擎: YOLOv8 ({self.config.pose_model})")
                device = "auto" if self.config.use_gpu else "cpu"
                detector = GPUOptimizedPoseDetector(
                    model_path=self.config.pose_model,
                    confidence=self.config.confidence,
                    device=device,
                    use_fp16=self.config.use_fp16
                )

            analyzer = None
            if self.config.enable_analysis:
                self.log_status("⚙ 初始化动作分析器...")
                analyzer = PoseAnalyzer()

            # 初始化装备检测器
            equipment_detector = None
            if self.config.enable_equipment_detection and SportsEquipmentDetector is not None:
                self.log_status("⚙ 初始化装备检测器（球拍和球）...")
                try:
                    equipment_detector = SportsEquipmentDetector(
                        model_path="yolov8n.pt",
                        conf_threshold=self.config.equipment_confidence,
                        use_gpu=self.config.use_gpu
                    )
                except Exception as e:
                    self.log_status(f"⚠ 装备检测器初始化失败: {e}")
                    equipment_detector = None

            # 打开视频
            cap = cv2.VideoCapture(input_video)
            if not cap.isOpened():
                raise Exception("无法打开输入视频")

            # 获取视频信息
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            self.log_status(f"📊 视频信息: {width}x{height} @ {fps}fps, {total_frames}帧")

            # 创建输出视频
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

            # 处理每一帧
            frame_count = 0
            pose_data_list = []
            start_time = time.time()

            while self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                progress = (frame_count / total_frames) * 100

                # 姿态检测 - 统一接口
                if self.config.pose_engine == "mediapipe":
                    # MediaPipe检测
                    keypoints, confidences = detector.detect(frame)

                    # 调试：记录检测结果
                    if frame_count == 1:
                        if keypoints is not None:
                            num_people = len(keypoints)
                            num_keypoints = keypoints.shape[1]
                            self.log_status(f"✓ MediaPipe检测到 {num_people} 个人，每人 {num_keypoints} 个关键点")
                        else:
                            self.log_status(f"⚠ MediaPipe未检测到任何人")

                    yolo_results = None  # MediaPipe不使用yolo_results
                else:
                    # YOLOv8检测
                    yolo_results = detector.model(frame, conf=self.config.confidence, verbose=False)

                    # 调试：记录检测结果
                    if frame_count == 1:
                        if yolo_results and len(yolo_results) > 0:
                            result = yolo_results[0]
                            if hasattr(result, 'keypoints') and result.keypoints is not None:
                                num_people = len(result.keypoints.xy)
                                if num_people > 0:
                                    num_keypoints = len(result.keypoints.xy[0])
                                    self.log_status(f"✓ YOLOv8检测到 {num_people} 个人，每人 {num_keypoints} 个关键点")
                                else:
                                    self.log_status(f"⚠ 检测到结果但没有关键点")
                            else:
                                self.log_status(f"⚠ 检测结果没有keypoints属性")
                        else:
                            self.log_status(f"⚠ 未检测到任何结果")

                    keypoints = None
                    confidences = None

                # 绘制姿态 - 使用自定义方法以支持细粒度控制
                try:
                    if self.config.pose_engine == "mediapipe":
                        # MediaPipe绘制
                        if keypoints is not None:
                            annotated_frame = detector.draw_skeleton(
                                frame.copy(),
                                keypoints,
                                confidences,
                                show_labels=self.config.show_labels
                            )
                        else:
                            annotated_frame = frame.copy()
                    else:
                        # YOLOv8绘制
                        if yolo_results and len(yolo_results) > 0:
                            # 先复制帧
                            annotated_frame = frame.copy()

                            # 绘制边界框和标签（使用YOLOv8）
                            if self.config.show_boxes or self.config.show_confidence or self.config.show_labels:
                                temp_frame = yolo_results[0].plot(
                                    line_width=self.config.line_width,
                                    font_size=0.5,
                                    boxes=self.config.show_boxes,
                                    conf=self.config.show_confidence,
                                    labels=self.config.show_labels,
                                    kpt_line=False  # 不绘制骨架，我们自己画
                                )
                                annotated_frame = temp_frame

                            # 自定义绘制骨架和关键点
                            annotated_frame = self._draw_pose_custom(
                                annotated_frame,
                                yolo_results,
                                draw_skeleton=self.config.show_skeleton,
                                draw_keypoints=self.config.show_keypoints
                            )
                        else:
                            annotated_frame = frame.copy()
                except Exception as e:
                    # 如果绘制失败，使用备用方法
                    self.log_status(f"⚠ 绘图失败: {e}")
                    if self.config.pose_engine == "mediapipe":
                        annotated_frame = frame.copy()
                    else:
                        annotated_frame = self._draw_pose_from_results(frame.copy(), yolo_results)

                # 装备检测和绘制
                if equipment_detector is not None:
                    try:
                        equipment_detections, _ = equipment_detector.detect(annotated_frame)

                        # 在第一帧记录检测结果
                        if frame_count == 1 and len(equipment_detections) > 0:
                            self.log_status(f"✓ 检测到 {len(equipment_detections)} 个装备")
                            for det in equipment_detections:
                                self.log_status(f"  - {det['class_name']}: {det['confidence']:.2f}")

                        # 绘制装备检测结果
                        if len(equipment_detections) > 0:
                            annotated_frame = equipment_detector.draw_detections(
                                annotated_frame,
                                equipment_detections,
                                draw_boxes=self.config.show_equipment_boxes,
                                draw_labels=self.config.show_equipment_labels,
                                draw_confidence=self.config.show_equipment_labels,  # 标签和置信度一起显示
                                box_color=(0, 255, 255),  # 黄色边框
                                text_color=(0, 0, 0),     # 黑色文字
                                line_width=self.config.line_width
                            )
                    except Exception as e:
                        if frame_count == 1:
                            self.log_status(f"⚠ 装备检测失败: {e}")

                # 调试：验证绘制是否成功
                if frame_count == 1:
                    # 对比原始帧和标注帧
                    import numpy as np
                    if np.array_equal(frame, annotated_frame):
                        self.log_status(f"✗ 第1帧标注失败（帧未改变）")
                    else:
                        self.log_status(f"✓ 第1帧标注完成（帧已改变）")

                # 分析姿态
                pose_info = None
                if analyzer:
                    if self.config.pose_engine == "mediapipe":
                        # MediaPipe分析
                        if keypoints is not None:
                            # MediaPipe返回的keypoints已经是numpy数组格式 (1, 17, 2)
                            kp = keypoints[0]  # 取第一个人
                            conf = confidences[0]
                            pose_info = analyzer.analyze_pose(kp, conf)
                            pose_data_list.append({
                                'frame': frame_count,
                                'pose_info': pose_info
                            })
                    else:
                        # YOLOv8分析
                        if yolo_results and len(yolo_results) > 0:
                            result = yolo_results[0]
                            if hasattr(result, 'keypoints') and result.keypoints is not None:
                                keypoints_data = result.keypoints.xy.cpu().numpy()
                                confidences_data = result.keypoints.conf.cpu().numpy()
                                if len(keypoints_data) > 0:
                                    kp = keypoints_data[0]
                                    conf = confidences_data[0]
                                    pose_info = analyzer.analyze_pose(kp, conf)
                                    pose_data_list.append({
                                        'frame': frame_count,
                                        'pose_info': pose_info
                                    })

                # 写入输出视频
                out.write(annotated_frame)

                # 更新进度和实时预览
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_count / elapsed if elapsed > 0 else 0
                    self.log_status(f"⏳ 处理进度: {progress:.1f}% ({frame_count}/{total_frames}) - {fps_actual:.1f} FPS")

                    # 实时显示当前帧（每10帧更新一次）
                    try:
                        self._update_preview_frames(frame, annotated_frame, pose_info)
                    except:
                        pass  # 忽略预览错误，继续处理

            # 清理资源
            cap.release()
            out.release()

            elapsed = time.time() - start_time
            self.log_status(f"✓ 处理完成! 用时: {elapsed:.1f}秒")
            self.log_status(f"  处理帧数: {frame_count}")
            self.log_status(f"  平均速度: {frame_count/elapsed:.1f} FPS")

            if pose_data_list:
                self.log_status(f"  分析数据: {len(pose_data_list)}帧")

                # 保存姿态分析数据到JSON文件
                pose_data_file = output_video.replace('.mp4', '_pose_data.json')
                try:
                    with open(pose_data_file, 'w', encoding='utf-8') as f:
                        json.dump(pose_data_list, f, ensure_ascii=False, indent=2)
                    self.log_status(f"✓ 姿态数据已保存: {Path(pose_data_file).name}")
                except Exception as e:
                    self.log_status(f"⚠ 姿态数据保存失败: {e}")

            # 复制音频到输出视频
            self.log_status("🔊 正在复制音频...")
            try:
                self._copy_audio(input_video, output_video)
                self.log_status("✓ 音频复制完成")
            except Exception as e:
                self.log_status(f"⚠ 音频复制失败: {e}")

            messagebox.showinfo("完成", f"视频处理完成！\n处理帧数: {frame_count}\n用时: {elapsed:.1f}秒")

            # 添加到历史记录
            self.add_to_history(input_video, output_video, pose_data_file if pose_data_list else None)

            # 加载处理后的视频和姿态数据
            self.player.load_processed_video(output_video)
            if pose_data_list:
                self.player.load_pose_data(pose_data_file)

        except Exception as e:
            self.log_status(f"✗ 处理失败: {e}")
            messagebox.showerror("错误", f"处理失败: {e}")
        finally:
            self.is_processing = False

    def _draw_pose_from_results(self, frame, yolo_results):
        """从YOLOv8 Results对象绘制姿态（备用方法）"""
        if not yolo_results or len(yolo_results) == 0:
            return frame

        result = yolo_results[0]
        if not hasattr(result, 'keypoints') or result.keypoints is None:
            return frame

        try:
            keypoints_data = result.keypoints.xy.cpu().numpy()
            confidences_data = result.keypoints.conf.cpu().numpy()

            if len(keypoints_data) == 0:
                return frame

            # 绘制第一个人的姿态
            keypoints = keypoints_data[0]
            confidences = confidences_data[0]

            return self._draw_pose_manual(frame, keypoints, confidences)
        except Exception as e:
            return frame

    def _draw_pose_custom(self, frame, yolo_results, draw_skeleton=True, draw_keypoints=True):
        """
        自定义绘制姿态，支持细粒度控制

        Args:
            frame: 要绘制的帧
            yolo_results: YOLOv8检测结果
            draw_skeleton: 是否绘制骨架线条
            draw_keypoints: 是否绘制关键点
        """
        if not yolo_results or len(yolo_results) == 0:
            return frame

        result = yolo_results[0]
        if not hasattr(result, 'keypoints') or result.keypoints is None:
            return frame

        try:
            # 提取关键点和置信度
            keypoints_data = result.keypoints.xy.cpu().numpy()
            confidences_data = result.keypoints.conf.cpu().numpy()

            if len(keypoints_data) == 0:
                return frame

            # 绘制第一个检测到的人
            keypoints = keypoints_data[0]
            confidences = confidences_data[0]

            # 定义骨架连接关系
            connections = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
                (5, 6),                           # 肩膀
                (5, 7), (7, 9),                   # 左臂
                (6, 8), (8, 10),                  # 右臂
                (5, 11), (6, 12),                 # 肩到臀
                (11, 12),                         # 臀部
                (11, 13), (13, 15),               # 左腿
                (12, 14), (14, 16),               # 右腿
            ]

            # 颜色映射函数
            def get_color_for_line(pt1_idx, pt2_idx):
                """根据连接线的两个关键点索引获取颜色（BGR格式）"""
                # 头部连接（0-4）
                if max(pt1_idx, pt2_idx) <= 4:
                    r, g, b = self.config.color_head[:3]
                    return (b, g, r)
                # 躯干连接（肩膀和臀部）
                elif (pt1_idx in [5, 6, 11, 12]) and (pt2_idx in [5, 6, 11, 12]):
                    r, g, b = self.config.color_torso[:3]
                    return (b, g, r)
                # 手臂连接（5-10）
                elif (5 <= pt1_idx <= 10) or (5 <= pt2_idx <= 10):
                    r, g, b = self.config.color_arms[:3]
                    return (b, g, r)
                # 腿部连接（11-16）
                else:
                    r, g, b = self.config.color_legs[:3]
                    return (b, g, r)

            def get_color_for_point(pt_idx):
                """根据关键点索引获取颜色（BGR格式）"""
                # 头部（0-4）
                if pt_idx <= 4:
                    r, g, b = self.config.color_head[:3]
                    return (b, g, r)
                # 手臂（5-10）
                elif 5 <= pt_idx <= 10:
                    r, g, b = self.config.color_arms[:3]
                    return (b, g, r)
                # 躯干（11-12）
                elif pt_idx in [11, 12]:
                    r, g, b = self.config.color_torso[:3]
                    return (b, g, r)
                # 腿部（13-16）
                else:
                    r, g, b = self.config.color_legs[:3]
                    return (b, g, r)

            # 绘制骨架线条
            if draw_skeleton:
                for pt1_idx, pt2_idx in connections:
                    if pt1_idx >= len(keypoints) or pt2_idx >= len(keypoints):
                        continue

                    pt1 = keypoints[pt1_idx]
                    pt2 = keypoints[pt2_idx]
                    conf1 = confidences[pt1_idx]
                    conf2 = confidences[pt2_idx]

                    if conf1 > self.config.confidence and conf2 > self.config.confidence:
                        # 根据连接线的两个端点获取颜色
                        color = get_color_for_line(pt1_idx, pt2_idx)
                        cv2.line(frame,
                                (int(pt1[0]), int(pt1[1])),
                                (int(pt2[0]), int(pt2[1])),
                                color,
                                self.config.line_width)

            # 绘制关键点
            if draw_keypoints:
                for i, (kp, conf) in enumerate(zip(keypoints, confidences)):
                    if conf > self.config.confidence:
                        # 根据关键点索引获取颜色
                        color = get_color_for_point(i)
                        cv2.circle(frame,
                                  (int(kp[0]), int(kp[1])),
                                  self.config.keypoint_size,
                                  color,
                                  -1)

            return frame
        except Exception as e:
            self.log_status(f"⚠ 自定义绘制失败: {e}")
            return frame

    def _draw_pose_manual(self, frame, keypoints, confidences):
        """手动绘制姿态（最底层方法）"""
        # 关键点连接关系
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
            (5, 6),                           # 肩膀
            (5, 7), (7, 9),                   # 左臂
            (6, 8), (8, 10),                  # 右臂
            (5, 11), (6, 12),                 # 肩到臀
            (11, 12),                         # 臀部
            (11, 13), (13, 15),               # 左腿
            (12, 14), (14, 16),               # 右腿
        ]

        # 颜色 (BGR格式)
        colors = {
            'head': (0, 0, 255),
            'arms': (0, 255, 0),
            'torso': (255, 0, 0),
            'legs': (0, 255, 255),
        }

        def get_color(pt1_idx, pt2_idx):
            if max(pt1_idx, pt2_idx) <= 4:
                return colors['head']
            elif (5 <= pt1_idx <= 10) or (5 <= pt2_idx <= 10):
                return colors['arms']
            elif (pt1_idx in [5, 6, 11, 12]) and (pt2_idx in [5, 6, 11, 12]):
                return colors['torso']
            else:
                return colors['legs']

        # 绘制连接线
        for pt1_idx, pt2_idx in connections:
            if pt1_idx >= len(keypoints) or pt2_idx >= len(keypoints):
                continue

            pt1 = keypoints[pt1_idx]
            pt2 = keypoints[pt2_idx]
            conf1 = confidences[pt1_idx]
            conf2 = confidences[pt2_idx]

            if conf1 > self.config.confidence and conf2 > self.config.confidence:
                color = get_color(pt1_idx, pt2_idx)
                cv2.line(frame,
                        (int(pt1[0]), int(pt1[1])),
                        (int(pt2[0]), int(pt2[1])),
                        color,
                        self.config.line_width)

        # 绘制关键点
        for i, (kp, conf) in enumerate(zip(keypoints, confidences)):
            if conf > self.config.confidence:
                if i <= 4:
                    color = colors['head']
                elif 5 <= i <= 10:
                    color = colors['arms']
                elif i in [5, 6, 11, 12]:
                    color = colors['torso']
                else:
                    color = colors['legs']

                cv2.circle(frame,
                          (int(kp[0]), int(kp[1])),
                          self.config.keypoint_size,
                          color,
                          -1)

        return frame

    def _draw_pose(self, frame, results):
        """绘制姿态骨架"""
        if not results or len(results) == 0:
            return frame

        result = results[0]
        if not hasattr(result, 'keypoints') or result.keypoints is None:
            return frame

        try:
            keypoints = result.keypoints.xy.cpu().numpy()[0]
            confidences = result.keypoints.conf.cpu().numpy()[0]
        except Exception as e:
            # 如果提取关键点失败，返回原始帧
            return frame

        # 验证关键点数量
        if len(keypoints) == 0:
            return frame

        # 关键点连接关系 (YOLOv8-pose 17个关键点)
        connections = [
            # 头部
            (0, 1), (0, 2),  # 鼻子到眼睛
            (1, 3), (2, 4),  # 眼睛到耳朵
            # 躯干
            (5, 6),   # 肩膀
            (5, 7), (7, 9),   # 左臂
            (6, 8), (8, 10),  # 右臂
            (5, 11), (6, 12),  # 肩到臀
            (11, 12),  # 臀部
            # 腿部
            (11, 13), (13, 15),  # 左腿
            (12, 14), (14, 16),  # 右腿
        ]

        # 定义身体部位颜色映射 (OpenCV使用BGR格式)
        def get_color_for_connection(pt1_idx, pt2_idx):
            """根据连接获取颜色 (BGR格式)"""
            # 头部 (0-4)
            if max(pt1_idx, pt2_idx) <= 4:
                r, g, b = self.config.color_head[:3]
                return (b, g, r)  # BGR
            # 手臂 (5-10)
            elif (5 <= pt1_idx <= 10) or (5 <= pt2_idx <= 10):
                if pt1_idx in [5, 7, 9] or pt2_idx in [5, 7, 9]:  # 左臂
                    r, g, b = self.config.color_arms[:3]
                    return (b, g, r)
                elif pt1_idx in [6, 8, 10] or pt2_idx in [6, 8, 10]:  # 右臂
                    r, g, b = self.config.color_arms[:3]
                    return (b, g, r)
            # 躯干 (5-6, 11-12)
            elif (pt1_idx in [5, 6, 11, 12]) and (pt2_idx in [5, 6, 11, 12]):
                r, g, b = self.config.color_torso[:3]
                return (b, g, r)
            # 腿部 (11-16)
            elif (11 <= pt1_idx <= 16) or (11 <= pt2_idx <= 16):
                r, g, b = self.config.color_legs[:3]
                return (b, g, r)

            # 默认颜色
            r, g, b = self.config.color_torso[:3]
            return (b, g, r)

        # 绘制骨架连接
        for pt1_idx, pt2_idx in connections:
            if pt1_idx >= len(keypoints) or pt2_idx >= len(keypoints):
                continue

            pt1 = keypoints[pt1_idx]
            pt2 = keypoints[pt2_idx]
            conf1 = confidences[pt1_idx]
            conf2 = confidences[pt2_idx]

            # 只有当两个关键点置信度都足够高时才绘制
            if conf1 > self.config.confidence and conf2 > self.config.confidence:
                color = get_color_for_connection(pt1_idx, pt2_idx)
                cv2.line(frame,
                        (int(pt1[0]), int(pt1[1])),
                        (int(pt2[0]), int(pt2[1])),
                        color,
                        self.config.line_width)

        # 绘制关键点
        for i, (kp, conf) in enumerate(zip(keypoints, confidences)):
            if conf > self.config.confidence:
                # 根据关键点索引选择颜色 (BGR格式)
                if i <= 4:  # 头部
                    r, g, b = self.config.color_head[:3]
                    color = (b, g, r)
                elif 5 <= i <= 10:  # 手臂
                    r, g, b = self.config.color_arms[:3]
                    color = (b, g, r)
                elif i in [5, 6, 11, 12]:  # 躯干
                    r, g, b = self.config.color_torso[:3]
                    color = (b, g, r)
                else:  # 腿部
                    r, g, b = self.config.color_legs[:3]
                    color = (b, g, r)

                cv2.circle(frame,
                          (int(kp[0]), int(kp[1])),
                          self.config.keypoint_size,
                          color,
                          -1)

                # 显示标签
                if self.config.show_labels:
                    label = f"{i}"
                    cv2.putText(frame, label,
                              (int(kp[0]) + 5, int(kp[1]) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.4, color, 1)

        return frame

    def _update_preview_frames(self, frame_orig, frame_proc, pose_info):
        """在处理过程中更新预览帧（在主线程中执行）"""
        def update_ui():
            try:
                self.player.display_frames(frame_orig, frame_proc, pose_info)
            except:
                pass

        # 使用after方法在主线程中更新UI
        self.root.after(0, update_ui)

    def _copy_audio(self, input_video: str, output_video: str):
        """复制音频到输出视频"""
        import subprocess

        # 创建临时文件名
        temp_output = output_video.replace('.mp4', '_temp.mp4')

        try:
            # 使用ffmpeg复制音频
            cmd = [
                'ffmpeg',
                '-i', output_video,      # 输入视频（无音频）
                '-i', input_video,       # 输入音频源
                '-c:v', 'copy',          # 复制视频流
                '-c:a', 'aac',           # 音频编码为AAC
                '-map', '0:v:0',         # 使用第一个输入的视频
                '-map', '1:a:0?',        # 使用第二个输入的音频（可选）
                '-shortest',             # 使用最短的流长度
                '-y',                    # 覆盖输出文件
                temp_output
            ]

            # 运行ffmpeg
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )

            if result.returncode == 0:
                # 成功：替换原文件
                import os
                if os.path.exists(output_video):
                    os.remove(output_video)
                os.rename(temp_output, output_video)
            else:
                # 失败：删除临时文件
                if Path(temp_output).exists():
                    Path(temp_output).unlink()
                raise Exception(f"ffmpeg返回错误: {result.returncode}")

        except FileNotFoundError:
            # ffmpeg未安装，使用moviepy作为备选
            self.log_status("⚠ 未找到ffmpeg，尝试使用moviepy...")
            try:
                from moviepy.editor import VideoFileClip, AudioFileClip

                # 加载视频和音频
                video_clip = VideoFileClip(output_video)
                audio_clip = AudioFileClip(input_video)

                # 合并音频
                final_clip = video_clip.set_audio(audio_clip)

                # 写入文件
                final_clip.write_videofile(
                    temp_output,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True,
                    logger=None
                )

                # 清理资源
                video_clip.close()
                audio_clip.close()
                final_clip.close()

                # 替换原文件
                import os
                if os.path.exists(output_video):
                    os.remove(output_video)
                os.rename(temp_output, output_video)

            except ImportError:
                self.log_status("⚠ moviepy未安装，跳过音频复制")
                self.log_status("提示：安装ffmpeg或moviepy以启用音频功能")

    def on_engine_changed(self, event=None):
        """检测引擎切换回调"""
        self.update_model_controls()

    def update_model_controls(self):
        """根据选择的引擎更新模型控件可见性"""
        engine = self.engine_var.get()

        if engine == "yolov8":
            # 显示YOLOv8控件
            self.model_label.grid()
            self.model_combo.grid()
            # 隐藏MediaPipe控件
            self.mediapipe_label.grid_remove()
            self.mediapipe_combo.grid_remove()
            self.mediapipe_info_label.grid_remove()
        else:  # mediapipe
            # 隐藏YOLOv8控件
            self.model_label.grid_remove()
            self.model_combo.grid_remove()
            # 显示MediaPipe控件
            self.mediapipe_label.grid()
            self.mediapipe_combo.grid()
            self.mediapipe_info_label.grid()

    def update_config_from_ui(self):
        """从UI更新配置"""
        self.config.pose_engine = self.engine_var.get()
        self.config.pose_model = self.model_var.get()
        self.config.mediapipe_complexity = self.mediapipe_var.get()
        self.config.skeleton_style = self.skeleton_style_var.get()
        self.config.confidence = self.confidence_var.get()
        self.config.use_gpu = self.gpu_var.get()
        self.config.use_fp16 = self.fp16_var.get()

        self.config.line_width = self.line_width_var.get()
        self.config.keypoint_size = self.keypoint_size_var.get()
        self.config.show_labels = self.show_labels_var.get()
        self.config.show_boxes = self.show_boxes_var.get()
        self.config.show_confidence = self.show_confidence_var.get()
        self.config.show_skeleton = self.show_skeleton_var.get()
        self.config.show_keypoints = self.show_keypoints_var.get()

        self.config.enable_equipment_detection = self.equipment_detect_var.get()
        self.config.show_equipment_boxes = self.equipment_boxes_var.get()
        self.config.show_equipment_labels = self.equipment_labels_var.get()
        self.config.equipment_confidence = self.equipment_conf_var.get()

        self.config.slowmo_factor = self.slowmo_var.get()
        self.config.video_quality = self.quality_var.get()

        self.config.color_head = self.color_vars["color_head"]
        self.config.color_arms = self.color_vars["color_arms"]
        self.config.color_torso = self.color_vars["color_torso"]
        self.config.color_legs = self.color_vars["color_legs"]

        self.config.enable_analysis = self.analysis_var.get()
        self.config.show_angles = self.show_angles_var.get()
        self.config.show_issues = self.show_issues_var.get()
        self.config.save_keyframes = self.save_frames_var.get()

    def save_config(self):
        """保存配置"""
        self.update_config_from_ui()

        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
            self.log_status(f"✓ 配置已保存: {self.config_file}")
            messagebox.showinfo("成功", "配置已保存")
        except Exception as e:
            self.log_status(f"✗ 保存配置失败: {e}")
            messagebox.showerror("错误", f"保存配置失败: {e}")

    def load_config(self):
        """加载配置"""
        if not self.config_file.exists():
            return

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.config = ProcessConfig.from_dict(data)

            # 更新UI
            self.engine_var.set(self.config.pose_engine)
            self.model_var.set(self.config.pose_model)
            self.mediapipe_var.set(self.config.mediapipe_complexity)
            self.skeleton_style_var.set(self.config.skeleton_style)
            self.confidence_var.set(self.config.confidence)
            self.gpu_var.set(self.config.use_gpu)
            self.fp16_var.set(self.config.use_fp16)

            # 更新模型控件可见性
            self.update_model_controls()

            self.line_width_var.set(self.config.line_width)
            self.keypoint_size_var.set(self.config.keypoint_size)
            self.show_labels_var.set(self.config.show_labels)
            self.show_boxes_var.set(self.config.show_boxes)
            self.show_confidence_var.set(self.config.show_confidence)
            self.slowmo_var.set(self.config.slowmo_factor)
            self.quality_var.set(self.config.video_quality)

            self.color_vars["color_head"] = self.config.color_head
            self.color_vars["color_arms"] = self.config.color_arms
            self.color_vars["color_torso"] = self.config.color_torso
            self.color_vars["color_legs"] = self.config.color_legs

            self.analysis_var.set(self.config.enable_analysis)
            self.show_angles_var.set(self.config.show_angles)
            self.show_issues_var.set(self.config.show_issues)
            self.save_frames_var.set(self.config.save_keyframes)

            self.log_status(f"✓ 配置已加载: {self.config_file}")
        except Exception as e:
            self.log_status(f"⚠ 加载配置失败: {e}")

    def reset_config(self):
        """重置配置"""
        self.config = ProcessConfig()
        self.load_config()  # 重新加载UI
        self.log_status("✓ 配置已重置为默认值")

    def load_processed_video(self):
        """加载已处理的视频"""
        from tkinter import filedialog

        # 选择已处理的视频文件
        processed_video = filedialog.askopenfilename(
            title="选择已处理的视频",
            filetypes=[("MP4文件", "*.mp4"), ("所有文件", "*.*")]
        )

        if not processed_video:
            return

        processed_path = Path(processed_video)

        # 尝试找到对应的原始视频和姿态数据文件
        # 假设文件命名规则：original.mp4 -> original_processed.mp4
        # 原始视频：去掉 _processed 后缀
        original_video = None
        if '_processed' in processed_path.stem:
            original_name = processed_path.stem.replace('_processed', '') + processed_path.suffix
            original_path = processed_path.parent / original_name
            if original_path.exists():
                original_video = str(original_path)

        # 姿态数据文件：同名但扩展名为 _pose_data.json
        pose_data_file = str(processed_path).replace('.mp4', '_pose_data.json')

        # 初始化播放器（如果还没有）
        if self.player is None:
            self.player = VideoComparePlayer(
                self.canvas_orig,
                self.canvas_proc,
                self.pose_canvas,
                self.frame_label
            )

        # 加载视频
        try:
            self.log_status(f"📂 加载已处理视频: {processed_path.name}")

            # 加载原始视频（如果找到）
            if original_video:
                self.player.load_original_video(original_video)
                self.log_status(f"  ✓ 原始视频: {Path(original_video).name}")
            else:
                self.log_status(f"  ⚠ 未找到原始视频")

            # 加载处理后的视频
            self.player.load_processed_video(processed_video)
            self.log_status(f"  ✓ 处理视频: {processed_path.name}")

            # 加载姿态数据（如果存在）
            if Path(pose_data_file).exists():
                self.player.load_pose_data(pose_data_file)
                self.log_status(f"  ✓ 姿态数据: {Path(pose_data_file).name}")
            else:
                self.log_status(f"  ⚠ 未找到姿态数据文件")

            # 更新进度条和帧标签
            self.update_progress_bar()

            # 显示第一帧
            frame_orig, frame_proc = self.player.get_current_frames()
            pose_data = self.player.pose_data_dict.get(1, None)
            self.player.display_frames(frame_orig, frame_proc, pose_data)

            self.log_status("✓ 视频加载完成，可以播放查看分析结果")

        except Exception as e:
            self.log_status(f"✗ 加载视频失败: {e}")
            messagebox.showerror("错误", f"加载视频失败: {e}")

    def update_progress_bar(self):
        """更新进度条范围和帧标签"""
        if self.player and self.player.total_frames > 0:
            # 重置进度条为0
            self.progress_var.set(0)
            # 更新帧标签
            self.frame_label.config(text=f"1 / {self.player.total_frames}")

    def add_to_history(self, original_video: str, processed_video: str, pose_data_file: str = None):
        """添加处理记录到历史"""
        import datetime

        # 创建历史记录项
        history_item = {
            'original_video': original_video,
            'processed_video': processed_video,
            'pose_data_file': pose_data_file,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'display_name': Path(processed_video).name
        }

        # 检查是否已存在（避免重复）
        for item in self.config.processing_history:
            if item.get('processed_video') == processed_video:
                # 已存在，更新时间戳
                item['timestamp'] = history_item['timestamp']
                self.save_config()
                self.refresh_history()
                return

        # 添加到列表开头（最新的在前面）
        self.config.processing_history.insert(0, history_item)

        # 限制历史记录数量（最多20条）
        if len(self.config.processing_history) > 20:
            self.config.processing_history = self.config.processing_history[:20]

        # 保存配置
        self.save_config()

        # 刷新历史记录下拉框
        self.refresh_history()

    def refresh_history(self):
        """刷新历史记录下拉框"""
        # 清理不存在的文件
        valid_history = []
        for item in self.config.processing_history:
            processed_video = item.get('processed_video', '')
            if processed_video and Path(processed_video).exists():
                valid_history.append(item)

        # 更新配置中的历史记录
        self.config.processing_history = valid_history

        # 如果清理了无效记录，保存配置
        if len(valid_history) < len(self.config.processing_history):
            self.save_config()

        # 更新下拉框选项
        display_names = [item['display_name'] for item in valid_history]
        self.history_combo['values'] = display_names

        # 如果有历史记录，设置提示文本
        if display_names:
            self.history_combo.set(f"选择历史记录 ({len(display_names)}条)")
        else:
            self.history_combo.set("暂无历史记录")

    def load_from_history(self, event=None):
        """从历史记录加载视频"""
        selected = self.history_var.get()

        # 查找对应的历史记录
        for item in self.config.processing_history:
            if item['display_name'] == selected:
                original_video = item.get('original_video')
                processed_video = item.get('processed_video')
                pose_data_file = item.get('pose_data_file')

                # 检查文件是否存在
                if not Path(processed_video).exists():
                    messagebox.showerror("错误", f"处理视频不存在:\n{processed_video}")
                    self.refresh_history()  # 清理无效记录
                    return

                # 初始化播放器（如果还没有）
                if self.player is None:
                    self.player = VideoComparePlayer(
                        self.canvas_original,
                        self.canvas_processed,
                        self.canvas_pose,
                        self.frame_label
                    )
                    self.player.update_callback = self.update_playback_progress

                try:
                    self.log_status(f"📂 从历史记录加载: {Path(processed_video).name}")

                    # 加载原始视频（如果存在）
                    if original_video and Path(original_video).exists():
                        self.player.load_original_video(original_video)
                        self.log_status(f"  ✓ 原始视频: {Path(original_video).name}")
                    else:
                        self.log_status(f"  ⚠ 原始视频不存在")

                    # 加载处理后的视频
                    self.player.load_processed_video(processed_video)
                    self.log_status(f"  ✓ 处理视频: {Path(processed_video).name}")

                    # 加载姿态数据（如果存在）
                    if pose_data_file and Path(pose_data_file).exists():
                        self.player.load_pose_data(pose_data_file)
                        self.log_status(f"  ✓ 姿态数据: {Path(pose_data_file).name}")
                    else:
                        self.log_status(f"  ⚠ 姿态数据不存在")

                    # 更新进度条和帧标签
                    self.update_progress_bar()

                    # 显示第一帧
                    frame_orig, frame_proc = self.player.get_current_frames()
                    pose_data = self.player.pose_data_dict.get(1, None)
                    self.player.display_frames(frame_orig, frame_proc, pose_data)

                    self.log_status("✓ 加载完成")

                except Exception as e:
                    self.log_status(f"✗ 加载失败: {e}")
                    messagebox.showerror("错误", f"加载失败: {e}")

                break

    def log_status(self, message: str):
        """记录状态"""
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()

    def on_closing(self):
        """关闭窗口"""
        if self.player:
            self.player.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = CompareGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
