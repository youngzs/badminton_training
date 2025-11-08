# -*- coding: utf-8 -*-
"""
羽毛球视频动作骨架识别程序 - GUI版本
带图形界面的视频处理工具
"""

import sys
import io
import os

# 设置Windows控制台输出编码为UTF-8
# 注意：在GUI程序（--windowed）中，stdout/stderr可能为None，需要安全处理
if sys.platform == 'win32':
    try:
        # 检查stdout是否存在且不为None（某些打包环境可能为None）
        if sys.stdout is not None and hasattr(sys.stdout, 'reconfigure'):
            try:
                sys.stdout.reconfigure(encoding='utf-8')
            except (AttributeError, ValueError, TypeError):
                # 如果不支持reconfigure，尝试使用buffer
                if hasattr(sys.stdout, 'buffer') and sys.stdout.buffer is not None:
                    try:
                        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
                    except Exception:
                        pass
        
        # 检查stderr是否存在且不为None
        if sys.stderr is not None and hasattr(sys.stderr, 'reconfigure'):
            try:
                sys.stderr.reconfigure(encoding='utf-8')
            except (AttributeError, ValueError, TypeError):
                # 如果不支持reconfigure，尝试使用buffer
                if hasattr(sys.stderr, 'buffer') and sys.stderr.buffer is not None:
                    try:
                        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
                    except Exception:
                        pass
    except Exception:
        # 忽略所有编码设置错误，GUI程序不需要控制台输出
        pass
    
    # 尝试设置控制台代码页（仅在控制台模式下有效）
    try:
        if sys.stdout is not None or sys.stderr is not None:
            os.system('chcp 65001 > nul 2>&1')
    except Exception:
        pass

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
import json

# 导入核心处理模块
import cv2
import numpy as np
import time
from pathlib import Path
from typing import Optional

try:
    from pose_detection_advanced import BadmintonAnalyzer, ObjectDetector, YOLOPoseDetector, MediaPipePoseDetector
except ImportError:
    # 如果导入失败，先尝试导入
    import importlib.util
    spec = importlib.util.spec_from_file_location("pose_detection_advanced", "pose_detection_advanced.py")
    if spec and spec.loader:
        pose_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pose_module)
        BadmintonAnalyzer = pose_module.BadmintonAnalyzer
        ObjectDetector = pose_module.ObjectDetector
        YOLOPoseDetector = pose_module.YOLOPoseDetector
        MediaPipePoseDetector = pose_module.MediaPipePoseDetector
    else:
        messagebox.showerror("Error", "Cannot import pose_detection_advanced module")
        sys.exit(1)


class VideoProcessorGUI:
    """视频处理GUI界面"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("羽毛球视频动作骨架识别工具")
        self.root.geometry("800x900")
        
        # 处理状态
        self.is_processing = False
        self.process_thread = None
        
        # 创建界面
        self.create_widgets()
        
        # 加载保存的配置
        self.load_config()
    
    def create_widgets(self):
        """创建界面组件"""
        
        # 主容器
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        row = 0
        
        # === 文件选择区域 ===
        file_frame = ttk.LabelFrame(main_frame, text="视频文件", padding="10")
        file_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        file_frame.columnconfigure(1, weight=1)
        row += 1
        
        ttk.Label(file_frame, text="输入视频:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.input_video_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.input_video_var, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(file_frame, text="浏览", command=self.browse_input_video).grid(row=0, column=2, padx=5)
        
        ttk.Label(file_frame, text="输出视频:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.output_video_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.output_video_var, width=50).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(file_frame, text="浏览", command=self.browse_output_video).grid(row=1, column=2, padx=5)
        
        # === 模型选择区域 ===
        model_frame = ttk.LabelFrame(main_frame, text="模型设置", padding="10")
        model_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        model_frame.columnconfigure(1, weight=1)
        row += 1
        
        # 姿态检测模型
        ttk.Label(model_frame, text="姿态检测模型:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.pose_model_var = tk.StringVar(value="yolo")
        pose_model_combo = ttk.Combobox(model_frame, textvariable=self.pose_model_var, 
                                       values=["yolo", "mediapipe"], state="readonly", width=15)
        pose_model_combo.grid(row=0, column=1, sticky=tk.W, padx=5)
        pose_model_combo.bind('<<ComboboxSelected>>', self.on_pose_model_change)
        
        ttk.Label(model_frame, text="模型大小:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.pose_model_size_var = tk.StringVar(value="n")
        ttk.Combobox(model_frame, textvariable=self.pose_model_size_var,
                     values=["n", "s", "m", "l", "x"], state="readonly", width=10).grid(row=0, column=3, padx=5)
        
        # 物体检测模型
        ttk.Label(model_frame, text="物体检测模型:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.object_model_size_var = tk.StringVar(value="n")
        ttk.Combobox(model_frame, textvariable=self.object_model_size_var,
                     values=["n", "s", "m", "l", "x"], state="readonly", width=15).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # === 检测参数区域 ===
        param_frame = ttk.LabelFrame(main_frame, text="检测参数", padding="10")
        param_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        param_frame.columnconfigure(1, weight=1)
        row += 1
        
        # 置信度阈值
        ttk.Label(param_frame, text="检测置信度:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.confidence_var = tk.DoubleVar(value=0.25)
        confidence_scale = ttk.Scale(param_frame, from_=0.1, to=0.5, variable=self.confidence_var,
                                     orient=tk.HORIZONTAL, length=200)
        confidence_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        self.confidence_label = ttk.Label(param_frame, text="0.25")
        self.confidence_label.grid(row=0, column=2, padx=5)
        confidence_scale.configure(command=lambda v: self.confidence_label.config(text=f"{float(v):.2f}"))
        
        # 球的置信度阈值
        ttk.Label(param_frame, text="球的置信度:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.ball_confidence_var = tk.DoubleVar(value=0.15)
        ball_confidence_scale = ttk.Scale(param_frame, from_=0.05, to=0.3, variable=self.ball_confidence_var,
                                          orient=tk.HORIZONTAL, length=200)
        ball_confidence_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        self.ball_confidence_label = ttk.Label(param_frame, text="0.15")
        self.ball_confidence_label.grid(row=1, column=2, padx=5)
        ball_confidence_scale.configure(command=lambda v: self.ball_confidence_label.config(text=f"{float(v):.2f}"))
        
        # 人物最小置信度
        ttk.Label(param_frame, text="人物最小置信度:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.person_min_conf_var = tk.DoubleVar(value=0.7)
        person_conf_scale = ttk.Scale(param_frame, from_=0.3, to=0.9, variable=self.person_min_conf_var,
                                      orient=tk.HORIZONTAL, length=200)
        person_conf_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5)
        self.person_conf_label = ttk.Label(param_frame, text="0.70")
        self.person_conf_label.grid(row=2, column=2, padx=5)
        person_conf_scale.configure(command=lambda v: self.person_conf_label.config(text=f"{float(v):.2f}"))
        
        # === 显示设置区域 ===
        display_frame = ttk.LabelFrame(main_frame, text="显示设置", padding="10")
        display_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        # 骨架线条宽度
        ttk.Label(display_frame, text="骨架线条宽度:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.line_width_var = tk.IntVar(value=1)
        line_width_scale = ttk.Scale(display_frame, from_=1, to=5, variable=self.line_width_var,
                                     orient=tk.HORIZONTAL, length=200)
        line_width_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        self.line_width_label = ttk.Label(display_frame, text="1")
        self.line_width_label.grid(row=0, column=2, padx=5)
        line_width_scale.configure(command=lambda v: self.line_width_label.config(text=f"{int(float(v))}"))
        
        # 关键点大小
        ttk.Label(display_frame, text="关键点大小:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.keypoint_size_var = tk.IntVar(value=3)
        keypoint_scale = ttk.Scale(display_frame, from_=2, to=8, variable=self.keypoint_size_var,
                                   orient=tk.HORIZONTAL, length=200)
        keypoint_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        self.keypoint_size_label = ttk.Label(display_frame, text="3")
        self.keypoint_size_label.grid(row=1, column=2, padx=5)
        keypoint_scale.configure(command=lambda v: self.keypoint_size_label.config(text=f"{int(float(v))}"))
        
        # 显示选项
        self.show_labels_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(display_frame, text="显示关键点标签", variable=self.show_labels_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        self.draw_person_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="绘制人物框", variable=self.draw_person_var).grid(row=2, column=2, sticky=tk.W, pady=5)
        
        # === 视频质量设置 ===
        quality_frame = ttk.LabelFrame(main_frame, text="视频质量控制", padding="10")
        quality_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        quality_frame.columnconfigure(1, weight=1)
        row += 1
        
        # 视频编码
        ttk.Label(quality_frame, text="视频编码:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.video_codec_var = tk.StringVar(value="mp4v")
        ttk.Combobox(quality_frame, textvariable=self.video_codec_var,
                     values=["mp4v", "XVID", "MJPG", "H264"], state="readonly", width=15).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # 音频处理
        self.keep_audio_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(quality_frame, text="保留原视频音频", variable=self.keep_audio_var).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # === 其他选项 ===
        option_frame = ttk.LabelFrame(main_frame, text="其他选项", padding="10")
        option_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        self.save_frames_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(option_frame, text="保存关键帧", variable=self.save_frames_var).grid(row=0, column=0, sticky=tk.W)
        
        self.show_preview_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(option_frame, text="显示预览窗口", variable=self.show_preview_var).grid(row=0, column=1, sticky=tk.W, padx=20)
        
        self.detect_objects_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(option_frame, text="检测物体（球、球拍）", variable=self.detect_objects_var).grid(row=1, column=0, sticky=tk.W, pady=5)
        
        # === 进度显示区域 ===
        progress_frame = ttk.LabelFrame(main_frame, text="处理进度", padding="10")
        progress_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        progress_frame.columnconfigure(0, weight=1)
        row += 1
        
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100, length=400)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5)
        
        self.progress_label = ttk.Label(progress_frame, text="等待开始...")
        self.progress_label.grid(row=1, column=0, pady=5)
        
        self.status_text = tk.Text(progress_frame, height=8, width=60, wrap=tk.WORD)
        self.status_text.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        scrollbar = ttk.Scrollbar(progress_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        scrollbar.grid(row=2, column=1, sticky=(tk.N, tk.S))
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
        # === 控制按钮 ===
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=2, pady=10)
        row += 1
        
        self.start_button = ttk.Button(button_frame, text="开始处理", command=self.start_processing, width=20)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="停止处理", command=self.stop_processing, width=20, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="保存配置", command=self.save_config, width=15).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="加载配置", command=self.load_config, width=15).pack(side=tk.LEFT, padx=5)
    
    def on_pose_model_change(self, event=None):
        """当姿态模型改变时的回调"""
        if self.pose_model_var.get() == "mediapipe":
            # MediaPipe不需要模型大小参数
            pass
    
    def browse_input_video(self):
        """浏览输入视频文件"""
        filename = filedialog.askopenfilename(
            title="选择输入视频",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")]
        )
        if filename:
            self.input_video_var.set(filename)
            # 自动设置输出文件名
            if not self.output_video_var.get():
                input_path = Path(filename)
                self.output_video_var.set(str(input_path.parent / f"{input_path.stem}_analysis.mp4"))
    
    def browse_output_video(self):
        """浏览输出视频文件"""
        filename = filedialog.asksaveasfilename(
            title="选择输出视频",
            defaultextension=".mp4",
            filetypes=[("MP4文件", "*.mp4"), ("所有文件", "*.*")]
        )
        if filename:
            self.output_video_var.set(filename)
    
    def log_status(self, message):
        """在状态文本框中记录日志"""
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
    
    def update_progress(self, value, message=""):
        """更新进度条"""
        self.progress_var.set(value)
        if message:
            self.progress_label.config(text=message)
        self.root.update_idletasks()
    
    def save_config(self):
        """保存配置到文件"""
        config = {
            "input_video": self.input_video_var.get(),
            "output_video": self.output_video_var.get(),
            "pose_model": self.pose_model_var.get(),
            "pose_model_size": self.pose_model_size_var.get(),
            "object_model_size": self.object_model_size_var.get(),
            "confidence": self.confidence_var.get(),
            "ball_confidence": self.ball_confidence_var.get(),
            "person_min_confidence": self.person_min_conf_var.get(),
            "line_width": self.line_width_var.get(),
            "keypoint_size": self.keypoint_size_var.get(),
            "show_labels": self.show_labels_var.get(),
            "draw_person": self.draw_person_var.get(),
            "video_codec": self.video_codec_var.get(),
            "keep_audio": self.keep_audio_var.get(),
            "save_frames": self.save_frames_var.get(),
            "show_preview": self.show_preview_var.get(),
            "detect_objects": self.detect_objects_var.get()
        }
        
        config_file = Path("gui_config.json")
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("成功", f"配置已保存到 {config_file}")
        except Exception as e:
            messagebox.showerror("错误", f"保存配置失败: {e}")
    
    def load_config(self):
        """从文件加载配置"""
        config_file = Path("gui_config.json")
        if not config_file.exists():
            return
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.input_video_var.set(config.get("input_video", ""))
            self.output_video_var.set(config.get("output_video", ""))
            self.pose_model_var.set(config.get("pose_model", "yolo"))
            self.pose_model_size_var.set(config.get("pose_model_size", "n"))
            self.object_model_size_var.set(config.get("object_model_size", "n"))
            self.confidence_var.set(config.get("confidence", 0.25))
            self.ball_confidence_var.set(config.get("ball_confidence", 0.15))
            self.person_min_conf_var.set(config.get("person_min_confidence", 0.7))
            self.line_width_var.set(config.get("line_width", 1))
            self.keypoint_size_var.set(config.get("keypoint_size", 3))
            self.show_labels_var.set(config.get("show_labels", False))
            self.draw_person_var.set(config.get("draw_person", True))
            self.video_codec_var.set(config.get("video_codec", "mp4v"))
            self.keep_audio_var.set(config.get("keep_audio", True))
            self.save_frames_var.set(config.get("save_frames", False))
            self.show_preview_var.set(config.get("show_preview", True))
            self.detect_objects_var.set(config.get("detect_objects", True))
            
            # 更新标签显示
            self.confidence_label.config(text=f"{self.confidence_var.get():.2f}")
            self.ball_confidence_label.config(text=f"{self.ball_confidence_var.get():.2f}")
            self.person_conf_label.config(text=f"{self.person_min_conf_var.get():.2f}")
            self.line_width_label.config(text=str(self.line_width_var.get()))
            self.keypoint_size_label.config(text=str(self.keypoint_size_var.get()))
            
        except Exception as e:
            messagebox.showwarning("警告", f"加载配置失败: {e}")
    
    def start_processing(self):
        """开始处理视频"""
        if self.is_processing:
            return
        
        # 验证输入
        input_video = self.input_video_var.get()
        if not input_video or not Path(input_video).exists():
            messagebox.showerror("错误", "请选择有效的输入视频文件")
            return
        
        output_video = self.output_video_var.get()
        if not output_video:
            messagebox.showerror("错误", "请指定输出视频文件")
            return
        
        # 更新UI状态
        self.is_processing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_text.delete(1.0, tk.END)
        self.progress_var.set(0.0)
        
        # 在新线程中处理
        self.process_thread = threading.Thread(target=self.process_video, daemon=True)
        self.process_thread.start()
    
    def stop_processing(self):
        """停止处理"""
        self.is_processing = False
        self.log_status("正在停止处理...")
        # 注意：实际停止需要在process_video中实现信号机制
    
    def process_video(self):
        """处理视频（在后台线程中运行）"""
        try:
            # 获取参数
            input_video = self.input_video_var.get()
            output_video = self.output_video_var.get()
            pose_model = self.pose_model_var.get()
            pose_model_size = self.pose_model_size_var.get()
            object_model_size = self.object_model_size_var.get()
            
            # 确定模型路径
            if pose_model == "yolo":
                pose_model_path = f"yolov8{pose_model_size}-pose.pt"
            else:
                pose_model_path = None
            
            object_model_path = f"yolov8{object_model_size}.pt"
            
            self.log_status(f"开始处理视频: {Path(input_video).name}")
            self.log_status(f"使用姿态模型: {pose_model} ({pose_model_size})")
            self.log_status(f"使用物体检测模型: {object_model_size}")
            
            # 创建自定义的分析器（支持线条宽度和关键点大小）
            analyzer = CustomBadmintonAnalyzer(
                pose_model=pose_model,
                pose_model_path=pose_model_path,
                object_model_path=object_model_path,
                confidence=self.confidence_var.get(),
                ball_confidence=self.ball_confidence_var.get(),
                line_width=self.line_width_var.get(),
                keypoint_size=self.keypoint_size_var.get(),
                video_codec=self.video_codec_var.get(),
                progress_callback=self.update_progress,
                status_callback=self.log_status,
                stop_flag=lambda: not self.is_processing
            )
            
            # 处理视频
            analyzer.process_video(
                video_path=input_video,
                output_path=output_video,
                show_preview=self.show_preview_var.get(),
                show_labels=self.show_labels_var.get(),
                save_frames=self.save_frames_var.get(),
                detect_objects=self.detect_objects_var.get(),
                draw_person=self.draw_person_var.get(),
                person_min_confidence=self.person_min_conf_var.get(),
                keep_audio=self.keep_audio_var.get()
            )
            
            if self.is_processing:
                self.log_status("处理完成！")
                self.update_progress(100, "处理完成")
                messagebox.showinfo("完成", "视频处理完成！")
            
        except Exception as e:
            self.log_status(f"错误: {e}")
            import traceback
            self.log_status(traceback.format_exc())
            messagebox.showerror("错误", f"处理失败: {e}")
        finally:
            # 恢复UI状态
            self.is_processing = False
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))


class CustomBadmintonAnalyzer(BadmintonAnalyzer):
    """自定义分析器，支持线条宽度和关键点大小设置"""
    
    def __init__(self, pose_model: str = "yolo", pose_model_path: str = "yolov8n-pose.pt",
                 object_model_path: str = "yolov8n.pt", confidence: float = 0.25,
                 ball_confidence: float = 0.15, line_width: int = 1, keypoint_size: int = 3,
                 video_codec: str = "mp4v", progress_callback=None, status_callback=None,
                 stop_flag=None):
        super().__init__(pose_model, pose_model_path, object_model_path, confidence, ball_confidence)
        self.line_width = line_width
        self.keypoint_size = keypoint_size
        self.video_codec = video_codec
        self.progress_callback = progress_callback
        self.status_callback = status_callback
        self.stop_flag = stop_flag
    
    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     show_preview: bool = True, show_labels: bool = False,
                     save_frames: bool = False, detect_objects: bool = True,
                     draw_person: bool = True, person_min_confidence: float = 0.7,
                     keep_audio: bool = True) -> None:
        """处理视频（带进度回调）"""
        import cv2
        import numpy as np
        import time
        from pathlib import Path
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if self.status_callback:
            self.status_callback(f"打开视频文件: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.status_callback:
            self.status_callback(f"视频信息: {width}x{height}, {fps}fps, 总帧数: {total_frames}")
        
        # 设置输出视频
        out = None
        temp_video_path = None
        if output_path:
            # 使用临时文件保存无音频的视频
            output_path_obj = Path(output_path)
            temp_video_path = str(output_path_obj.parent / f"{output_path_obj.stem}_temp{output_path_obj.suffix}")
            fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
            if self.status_callback:
                self.status_callback(f"输出视频将保存到: {output_path}")
                self.status_callback(f"临时视频文件: {temp_video_path}")
        
        # 创建关键帧保存目录
        frames_dir = None
        if save_frames:
            frames_dir = video_path.parent / f"{video_path.stem}_keyframes"
            frames_dir.mkdir(exist_ok=True)
            if self.status_callback:
                self.status_callback(f"关键帧将保存到: {frames_dir}")
        
        frame_count = 0
        keyframe_count = 0
        last_person_count = 0
        total_pose_time = 0
        total_object_time = 0
        
        if self.status_callback:
            self.status_callback("开始处理视频...")
        
        try:
            while True:
                if self.stop_flag and self.stop_flag():
                    if self.status_callback:
                        self.status_callback("用户中断处理")
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 姿态检测
                start_time = time.time()
                keypoints_list, confidences_list = self.pose_detector.detect(frame)
                pose_time = time.time() - start_time
                total_pose_time += pose_time
                
                # 物体检测
                object_detections = []
                if detect_objects:
                    start_time = time.time()
                    object_detections = self.object_detector.detect(frame)
                    object_time = time.time() - start_time
                    total_object_time += object_time
                
                # 绘制骨架（使用自定义线条宽度和关键点大小）
                if keypoints_list:
                    keypoints_array = np.array(keypoints_list)
                    confidences_array = np.array(confidences_list)
                    frame = self.draw_skeleton_custom(frame, keypoints_array, confidences_array, 
                                                     show_labels, self.line_width, self.keypoint_size)
                
                # 绘制物体检测
                if object_detections:
                    frame = self.object_detector.draw_detections(frame, object_detections,
                                                               draw_person=draw_person,
                                                               person_min_confidence=person_min_confidence,
                                                               show_labels=False)
                
                # 保存关键帧
                person_count = len(keypoints_list)
                if save_frames and person_count != last_person_count:
                    keyframe_path = frames_dir / f"keyframe_{keyframe_count:04d}_frame_{frame_count}.jpg"
                    cv2.imwrite(str(keyframe_path), frame)
                    keyframe_count += 1
                    if self.status_callback:
                        self.status_callback(f"保存关键帧: {keyframe_path}")
                
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
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        if self.status_callback:
                            self.status_callback("用户中断处理")
                        break
                
                # 更新进度
                if total_frames > 0:
                    progress = (frame_count / total_frames) * 100
                    if self.progress_callback:
                        self.progress_callback(progress, f"处理中: {frame_count}/{total_frames} 帧")
                    
                    if frame_count % 30 == 0 and self.status_callback:
                        avg_pose_fps = frame_count / total_pose_time if total_pose_time > 0 else 0
                        avg_object_fps = frame_count / total_object_time if total_object_time > 0 else 0
                        self.status_callback(f"进度: {progress:.1f}% | 姿态FPS: {avg_pose_fps:.1f} | 物体FPS: {avg_object_fps:.1f}")
        
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            # 合并音频
            if output_path and temp_video_path and Path(temp_video_path).exists() and keep_audio:
                if self.status_callback:
                    self.status_callback("正在合并音频...")
                try:
                    self._merge_audio(str(video_path), temp_video_path, output_path)
                    # 删除临时文件
                    Path(temp_video_path).unlink()
                    if self.status_callback:
                        self.status_callback(f"输出视频已保存: {output_path}")
                except Exception as e:
                    if self.status_callback:
                        self.status_callback(f"警告: 无法合并音频: {e}")
                    # 如果合并失败，使用临时文件作为输出
                    import shutil
                    shutil.move(temp_video_path, output_path)
            
            if self.status_callback:
                self.status_callback(f"处理完成！总帧数: {frame_count}")
                if save_frames:
                    self.status_callback(f"保存关键帧数: {keyframe_count}")
                if total_pose_time > 0:
                    self.status_callback(f"平均姿态检测FPS: {frame_count / total_pose_time:.2f}")
                if total_object_time > 0:
                    self.status_callback(f"平均物体检测FPS: {frame_count / total_object_time:.2f}")
    
    def draw_skeleton_custom(self, image: np.ndarray, keypoints: np.ndarray,
                            confidences: np.ndarray, show_labels: bool = False,
                            line_width: int = 1, keypoint_size: int = 3) -> np.ndarray:
        """绘制骨架（自定义线条宽度和关键点大小）"""
        result_image = image.copy()
        colors = [
            (255, 0, 0),    # 红色 - 头部
            (0, 255, 0),    # 绿色 - 手臂
            (0, 0, 255),    # 蓝色 - 躯干
            (255, 255, 0),  # 青色 - 腿部
        ]
        
        # 获取骨架连接和关键点名称
        skeleton_connections = self.pose_detector.SKELETON_CONNECTIONS
        keypoint_names = self.pose_detector.KEYPOINT_NAMES
        
        for person_idx, (kp, conf) in enumerate(zip(keypoints, confidences)):
            # 绘制关键点
            for i, (point, conf_score) in enumerate(zip(kp, conf)):
                if conf_score > 0.5:
                    x, y = int(point[0]), int(point[1])
                    if i < 5:
                        color = colors[0]
                        radius = keypoint_size
                    elif i < 11:
                        color = colors[1]
                        radius = keypoint_size
                    elif i < 13:
                        color = colors[2]
                        radius = keypoint_size
                    else:
                        color = colors[3]
                        radius = keypoint_size
                    
                    cv2.circle(result_image, (x, y), radius, color, -1)
                    
                    if show_labels and i < len(keypoint_names):
                        cv2.putText(result_image, keypoint_names[i],
                                  (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.4, color, 1)
            
            # 绘制骨架连接（使用自定义线条宽度）
            for connection in skeleton_connections:
                idx1, idx2 = connection
                if idx1 < len(conf) and idx2 < len(conf) and conf[idx1] > 0.5 and conf[idx2] > 0.5:
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
                    
                    cv2.line(result_image, pt1, pt2, color, line_width)
        
        return result_image
    
    def _merge_audio(self, input_video: str, temp_video: str, output_video: str) -> None:
        """合并原视频的音频到输出视频"""
        import subprocess
        import shutil
        
        # 首先尝试使用moviepy（更可靠）
        try:
            from moviepy.editor import VideoFileClip
            
            if self.status_callback:
                self.status_callback("使用moviepy合并音频...")
            
            # 加载视频和音频
            video_clip = VideoFileClip(temp_video)
            
            # 检查原视频是否有音频
            input_clip = VideoFileClip(input_video)
            if input_clip.audio is None:
                if self.status_callback:
                    self.status_callback("警告: 原视频没有音频轨道")
                # 如果没有音频，直接复制视频
                video_clip.write_videofile(output_video, codec='libx264', audio=False,
                                         remove_temp=True, verbose=False, logger=None)
                video_clip.close()
                input_clip.close()
                return
            
            audio_clip = input_clip.audio
            
            # 合并音频
            final_clip = video_clip.set_audio(audio_clip)
            
            # 写入新文件
            final_clip.write_videofile(output_video, codec='libx264', audio_codec='aac',
                                     remove_temp=True, verbose=False, logger=None)
            
            # 清理
            video_clip.close()
            audio_clip.close()
            input_clip.close()
            final_clip.close()
            
            if self.status_callback:
                self.status_callback("音频合并成功")
            return
            
        except ImportError:
            if self.status_callback:
                self.status_callback("moviepy未安装，尝试使用ffmpeg...")
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"moviepy合并失败: {e}，尝试使用ffmpeg...")
        
        # 如果moviepy失败，尝试使用ffmpeg
        try:
            if self.status_callback:
                self.status_callback("使用ffmpeg合并音频...")
            
            # 使用ffmpeg合并音频
            cmd = [
                'ffmpeg',
                '-y',  # 覆盖输出文件
                '-i', temp_video,  # 输入视频（无音频）
                '-i', input_video,  # 原视频（有音频）
                '-c:v', 'copy',  # 视频编码：直接复制
                '-c:a', 'aac',  # 音频编码：使用AAC
                '-map', '0:v:0',  # 使用第一个输入的视频流
                '-map', '1:a:0',  # 使用第二个输入的音频流
                '-shortest',  # 以最短的流为准
                output_video
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if self.status_callback:
                self.status_callback("ffmpeg音频合并成功")
            
        except subprocess.CalledProcessError as e:
            error_msg = f"ffmpeg合并失败: {e.stderr}"
            if self.status_callback:
                self.status_callback(error_msg)
            raise RuntimeError(error_msg)
        except FileNotFoundError:
            error_msg = "ffmpeg未找到，请安装ffmpeg或moviepy"
            if self.status_callback:
                self.status_callback(error_msg)
            raise RuntimeError(error_msg)


def main():
    """主函数"""
    root = tk.Tk()
    app = VideoProcessorGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()

