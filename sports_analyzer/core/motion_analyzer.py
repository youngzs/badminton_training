"""
核心动作分析模块
使用MediaPipe和自定义算法进行运动姿势分析
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math


class SportType(Enum):
    BADMINTON = "badminton"
    TENNIS = "tennis"
    BASKETBALL = "basketball"
    GOLF = "golf"
    RUNNING = "running"
    YOGA = "yoga"


@dataclass
class JointAngle:
    joint_name: str
    angle: float
    optimal_range: Tuple[float, float]
    deviation: float


@dataclass
class MotionFrame:
    timestamp: float
    landmarks: np.ndarray
    joint_angles: List[JointAngle]
    score: float
    issues: List[str]


class MotionAnalyzer:
    """运动动作分析器"""
    
    def __init__(self, sport_type: SportType = SportType.BADMINTON):
        self.sport_type = sport_type
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.frame_history: List[MotionFrame] = []
        self.calibration_data = None
        
    def process_frame(self, frame: np.ndarray, timestamp: float) -> MotionFrame:
        """处理单帧图像"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
            
        landmarks = self._extract_landmarks(results.pose_landmarks)
        joint_angles = self._calculate_joint_angles(landmarks)
        score, issues = self._evaluate_pose(landmarks, joint_angles)
        
        motion_frame = MotionFrame(
            timestamp=timestamp,
            landmarks=landmarks,
            joint_angles=joint_angles,
            score=score,
            issues=issues
        )
        
        self.frame_history.append(motion_frame)
        if len(self.frame_history) > 300:  # 保留最近10秒@30fps
            self.frame_history.pop(0)
            
        return motion_frame
    
    def _extract_landmarks(self, pose_landmarks) -> np.ndarray:
        """提取关键点坐标"""
        landmarks = []
        for landmark in pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        return np.array(landmarks)
    
    def _calculate_joint_angles(self, landmarks: np.ndarray) -> List[JointAngle]:
        """计算关节角度"""
        angles = []
        
        # 定义关节连接关系
        joint_connections = {
            'left_elbow': (11, 13, 15),  # shoulder-elbow-wrist
            'right_elbow': (12, 14, 16),
            'left_shoulder': (13, 11, 23),  # elbow-shoulder-hip
            'right_shoulder': (14, 12, 24),
            'left_knee': (23, 25, 27),  # hip-knee-ankle
            'right_knee': (24, 26, 28),
            'left_hip': (11, 23, 25),  # shoulder-hip-knee
            'right_hip': (12, 24, 26),
        }
        
        # 定义各运动类型的最优角度范围
        optimal_angles = self._get_optimal_angles()
        
        for joint_name, (p1, p2, p3) in joint_connections.items():
            angle = self._calculate_angle(
                landmarks[p1][:3],
                landmarks[p2][:3],
                landmarks[p3][:3]
            )
            
            optimal_range = optimal_angles.get(joint_name, (0, 180))
            deviation = self._calculate_deviation(angle, optimal_range)
            
            angles.append(JointAngle(
                joint_name=joint_name,
                angle=angle,
                optimal_range=optimal_range,
                deviation=deviation
            ))
            
        return angles
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """计算三点形成的角度"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _calculate_deviation(self, angle: float, optimal_range: Tuple[float, float]) -> float:
        """计算角度偏差"""
        if optimal_range[0] <= angle <= optimal_range[1]:
            return 0.0
        elif angle < optimal_range[0]:
            return optimal_range[0] - angle
        else:
            return angle - optimal_range[1]
    
    def _get_optimal_angles(self) -> Dict[str, Tuple[float, float]]:
        """获取不同运动的最优角度范围"""
        if self.sport_type == SportType.BADMINTON:
            return {
                'left_elbow': (90, 150),
                'right_elbow': (90, 150),
                'left_shoulder': (70, 120),
                'right_shoulder': (70, 120),
                'left_knee': (120, 170),
                'right_knee': (120, 170),
                'left_hip': (90, 150),
                'right_hip': (90, 150),
            }
        elif self.sport_type == SportType.TENNIS:
            return {
                'left_elbow': (80, 140),
                'right_elbow': (80, 140),
                'left_shoulder': (60, 110),
                'right_shoulder': (60, 110),
                'left_knee': (110, 160),
                'right_knee': (110, 160),
                'left_hip': (85, 145),
                'right_hip': (85, 145),
            }
        else:
            # 默认值
            return {joint: (45, 135) for joint in [
                'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
                'left_knee', 'right_knee', 'left_hip', 'right_hip'
            ]}
    
    def _evaluate_pose(self, landmarks: np.ndarray, joint_angles: List[JointAngle]) -> Tuple[float, List[str]]:
        """评估姿势质量"""
        issues = []
        scores = []
        
        # 1. 关节角度评分 (40%)
        angle_score = self._evaluate_angles(joint_angles, issues)
        scores.append(angle_score * 0.4)
        
        # 2. 身体平衡评分 (30%)
        balance_score = self._evaluate_balance(landmarks, issues)
        scores.append(balance_score * 0.3)
        
        # 3. 动作流畅度评分 (20%)
        fluidity_score = self._evaluate_fluidity(landmarks, issues)
        scores.append(fluidity_score * 0.2)
        
        # 4. 对称性评分 (10%)
        symmetry_score = self._evaluate_symmetry(landmarks, joint_angles, issues)
        scores.append(symmetry_score * 0.1)
        
        total_score = sum(scores)
        return total_score, issues
    
    def _evaluate_angles(self, joint_angles: List[JointAngle], issues: List[str]) -> float:
        """评估关节角度"""
        if not joint_angles:
            return 0.0
            
        scores = []
        for angle in joint_angles:
            if angle.deviation == 0:
                scores.append(1.0)
            elif angle.deviation < 10:
                scores.append(0.9)
            elif angle.deviation < 20:
                scores.append(0.7)
                issues.append(f"{angle.joint_name}角度略有偏差({angle.deviation:.1f}°)")
            elif angle.deviation < 30:
                scores.append(0.5)
                issues.append(f"{angle.joint_name}角度偏差较大({angle.deviation:.1f}°)")
            else:
                scores.append(0.2)
                issues.append(f"{angle.joint_name}角度严重偏差({angle.deviation:.1f}°)")
                
        return np.mean(scores) if scores else 0.0
    
    def _evaluate_balance(self, landmarks: np.ndarray, issues: List[str]) -> float:
        """评估身体平衡"""
        # 计算重心位置
        hip_center = (landmarks[23][:2] + landmarks[24][:2]) / 2
        ankle_center = (landmarks[27][:2] + landmarks[28][:2]) / 2
        
        # 重心偏移量
        offset = np.linalg.norm(hip_center - ankle_center)
        
        if offset < 0.1:
            return 1.0
        elif offset < 0.2:
            return 0.8
        elif offset < 0.3:
            issues.append("身体重心略有偏移")
            return 0.6
        else:
            issues.append("身体平衡性不佳")
            return 0.4
    
    def _evaluate_fluidity(self, landmarks: np.ndarray, issues: List[str]) -> float:
        """评估动作流畅度"""
        if len(self.frame_history) < 10:
            return 1.0  # 数据不足，给默认分
            
        # 计算关键点速度变化
        recent_frames = self.frame_history[-10:]
        velocities = []
        
        for i in range(1, len(recent_frames)):
            prev = recent_frames[i-1].landmarks
            curr = recent_frames[i].landmarks
            dt = recent_frames[i].timestamp - recent_frames[i-1].timestamp
            
            if dt > 0:
                velocity = np.linalg.norm(curr - prev) / dt
                velocities.append(velocity)
        
        if not velocities:
            return 1.0
            
        # 计算速度变化的标准差
        velocity_std = np.std(velocities)
        
        if velocity_std < 0.5:
            return 1.0
        elif velocity_std < 1.0:
            return 0.8
        elif velocity_std < 2.0:
            issues.append("动作流畅度有待提高")
            return 0.6
        else:
            issues.append("动作不够流畅")
            return 0.4
    
    def _evaluate_symmetry(self, landmarks: np.ndarray, joint_angles: List[JointAngle], issues: List[str]) -> float:
        """评估左右对称性"""
        symmetry_pairs = [
            ('left_elbow', 'right_elbow'),
            ('left_shoulder', 'right_shoulder'),
            ('left_knee', 'right_knee'),
            ('left_hip', 'right_hip')
        ]
        
        angle_dict = {angle.joint_name: angle.angle for angle in joint_angles}
        differences = []
        
        for left, right in symmetry_pairs:
            if left in angle_dict and right in angle_dict:
                diff = abs(angle_dict[left] - angle_dict[right])
                differences.append(diff)
        
        if not differences:
            return 1.0
            
        avg_diff = np.mean(differences)
        
        if avg_diff < 10:
            return 1.0
        elif avg_diff < 20:
            return 0.8
        elif avg_diff < 30:
            issues.append("左右动作对称性需要改善")
            return 0.6
        else:
            issues.append("左右动作不对称")
            return 0.4
    
    def get_movement_distance(self) -> float:
        """计算移动距离"""
        if len(self.frame_history) < 2:
            return 0.0
            
        total_distance = 0.0
        for i in range(1, len(self.frame_history)):
            prev_hip = (self.frame_history[i-1].landmarks[23][:2] + 
                       self.frame_history[i-1].landmarks[24][:2]) / 2
            curr_hip = (self.frame_history[i].landmarks[23][:2] + 
                       self.frame_history[i].landmarks[24][:2]) / 2
            
            distance = np.linalg.norm(curr_hip - prev_hip)
            total_distance += distance
            
        return total_distance
    
    def get_speed_metrics(self) -> Dict[str, float]:
        """获取速度指标"""
        if len(self.frame_history) < 2:
            return {'avg_speed': 0.0, 'max_speed': 0.0}
            
        speeds = []
        for i in range(1, len(self.frame_history)):
            prev = self.frame_history[i-1]
            curr = self.frame_history[i]
            dt = curr.timestamp - prev.timestamp
            
            if dt > 0:
                prev_hip = (prev.landmarks[23][:2] + prev.landmarks[24][:2]) / 2
                curr_hip = (curr.landmarks[23][:2] + curr.landmarks[24][:2]) / 2
                speed = np.linalg.norm(curr_hip - prev_hip) / dt
                speeds.append(speed)
        
        if speeds:
            return {
                'avg_speed': np.mean(speeds),
                'max_speed': np.max(speeds)
            }
        return {'avg_speed': 0.0, 'max_speed': 0.0}
    
    def draw_skeleton(self, frame: np.ndarray, motion_frame: MotionFrame) -> np.ndarray:
        """在图像上绘制骨架"""
        if motion_frame is None:
            return frame
            
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # 绘制关键点
        for i, landmark in enumerate(motion_frame.landmarks):
            x = int(landmark[0] * w)
            y = int(landmark[1] * h)
            cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)
        
        # 绘制骨架连接
        connections = self.mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start = motion_frame.landmarks[start_idx]
            end = motion_frame.landmarks[end_idx]
            
            start_point = (int(start[0] * w), int(start[1] * h))
            end_point = (int(end[0] * w), int(end[1] * h))
            
            cv2.line(annotated_frame, start_point, end_point, (0, 255, 0), 2)
        
        # 显示分数和问题
        score_text = f"Score: {motion_frame.score:.2f}"
        cv2.putText(annotated_frame, score_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示问题提示
        y_offset = 60
        for issue in motion_frame.issues[:3]:  # 最多显示3个问题
            cv2.putText(annotated_frame, issue, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            y_offset += 25
            
        return annotated_frame
    
    def reset(self):
        """重置分析器状态"""
        self.frame_history.clear()
        self.calibration_data = None
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'pose'):
            self.pose.close()