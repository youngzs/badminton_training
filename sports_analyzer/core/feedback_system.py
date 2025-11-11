"""
智能反馈系统
根据动作分析结果生成个性化训练建议
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from .motion_analyzer import MotionFrame, JointAngle, SportType


class FeedbackLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    NEEDS_IMPROVEMENT = "needs_improvement"


class FeedbackType(Enum):
    POSTURE = "posture"
    TIMING = "timing"
    BALANCE = "balance"
    COORDINATION = "coordination"
    STRENGTH = "strength"
    FLEXIBILITY = "flexibility"


@dataclass
class Suggestion:
    type: FeedbackType
    priority: int  # 1-5, 1最高
    title: str
    description: str
    visual_hint: Optional[str] = None
    drill_recommendation: Optional[str] = None


@dataclass
class TrainingFeedback:
    overall_score: float
    level: FeedbackLevel
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[Suggestion]
    progress_notes: Optional[str] = None


class FeedbackSystem:
    """智能反馈生成系统"""
    
    def __init__(self, sport_type: SportType = SportType.BADMINTON):
        self.sport_type = sport_type
        self.feedback_history: List[TrainingFeedback] = []
        self.user_skill_level = "intermediate"  # 可以动态调整
        
    def generate_feedback(self, motion_frames: List[MotionFrame]) -> TrainingFeedback:
        """生成综合训练反馈"""
        if not motion_frames:
            return self._generate_empty_feedback()
            
        # 计算整体评分
        overall_score = self._calculate_overall_score(motion_frames)
        level = self._determine_level(overall_score)
        
        # 分析优缺点
        strengths = self._analyze_strengths(motion_frames)
        weaknesses = self._analyze_weaknesses(motion_frames)
        
        # 生成改进建议
        suggestions = self._generate_suggestions(motion_frames, weaknesses)
        
        # 生成进度说明
        progress_notes = self._generate_progress_notes(overall_score)
        
        feedback = TrainingFeedback(
            overall_score=overall_score,
            level=level,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            progress_notes=progress_notes
        )
        
        self.feedback_history.append(feedback)
        return feedback
    
    def _calculate_overall_score(self, motion_frames: List[MotionFrame]) -> float:
        """计算整体分数"""
        if not motion_frames:
            return 0.0
            
        scores = [frame.score for frame in motion_frames if frame.score > 0]
        if not scores:
            return 0.0
            
        # 加权平均：最近的帧权重更高
        weights = np.linspace(0.5, 1.0, len(scores))
        weighted_score = np.average(scores, weights=weights)
        
        return min(max(weighted_score * 100, 0), 100)
    
    def _determine_level(self, score: float) -> FeedbackLevel:
        """确定表现级别"""
        if score >= 90:
            return FeedbackLevel.EXCELLENT
        elif score >= 75:
            return FeedbackLevel.GOOD
        elif score >= 60:
            return FeedbackLevel.FAIR
        else:
            return FeedbackLevel.NEEDS_IMPROVEMENT
    
    def _analyze_strengths(self, motion_frames: List[MotionFrame]) -> List[str]:
        """分析动作优点"""
        strengths = []
        
        # 统计各项指标
        angle_scores = []
        balance_scores = []
        consistency_scores = []
        
        for frame in motion_frames:
            if frame.joint_angles:
                # 角度准确性
                angle_deviations = [angle.deviation for angle in frame.joint_angles]
                angle_score = 1.0 - (np.mean(angle_deviations) / 45.0)  # 标准化到0-1
                angle_scores.append(angle_score)
                
        # 分析一致性
        if len(motion_frames) > 10:
            scores = [f.score for f in motion_frames[-10:]]
            consistency = 1.0 - np.std(scores)
            consistency_scores.append(consistency)
        
        # 生成优点描述
        if angle_scores and np.mean(angle_scores) > 0.8:
            strengths.append("关节角度控制准确")
            
        if consistency_scores and np.mean(consistency_scores) > 0.7:
            strengths.append("动作稳定性良好")
            
        # 检查特定动作的优点
        if self.sport_type == SportType.BADMINTON:
            strengths.extend(self._analyze_badminton_strengths(motion_frames))
        elif self.sport_type == SportType.TENNIS:
            strengths.extend(self._analyze_tennis_strengths(motion_frames))
            
        return strengths[:5]  # 最多返回5个优点
    
    def _analyze_weaknesses(self, motion_frames: List[MotionFrame]) -> List[str]:
        """分析动作缺点"""
        weaknesses = []
        issue_counter = {}
        
        # 统计所有问题
        for frame in motion_frames:
            for issue in frame.issues:
                # 提取问题类型
                issue_type = self._categorize_issue(issue)
                if issue_type not in issue_counter:
                    issue_counter[issue_type] = 0
                issue_counter[issue_type] += 1
        
        # 找出最常见的问题
        total_frames = len(motion_frames)
        for issue_type, count in issue_counter.items():
            frequency = count / total_frames if total_frames > 0 else 0
            if frequency > 0.3:  # 30%以上的帧出现该问题
                weaknesses.append(issue_type)
        
        # 添加整体性问题分析
        avg_score = np.mean([f.score for f in motion_frames])
        if avg_score < 0.6:
            weaknesses.append("整体动作协调性需要加强")
            
        return weaknesses[:5]  # 最多返回5个缺点
    
    def _categorize_issue(self, issue: str) -> str:
        """问题分类"""
        if "角度" in issue or "偏差" in issue:
            return "关节角度控制不当"
        elif "平衡" in issue or "重心" in issue:
            return "身体平衡性不足"
        elif "流畅" in issue:
            return "动作流畅度欠佳"
        elif "对称" in issue:
            return "左右动作不对称"
        else:
            return issue
    
    def _generate_suggestions(self, motion_frames: List[MotionFrame], weaknesses: List[str]) -> List[Suggestion]:
        """生成改进建议"""
        suggestions = []
        
        for weakness in weaknesses:
            if "角度" in weakness:
                suggestions.append(self._create_angle_suggestion())
            elif "平衡" in weakness:
                suggestions.append(self._create_balance_suggestion())
            elif "流畅" in weakness:
                suggestions.append(self._create_fluidity_suggestion())
            elif "对称" in weakness:
                suggestions.append(self._create_symmetry_suggestion())
            elif "协调" in weakness:
                suggestions.append(self._create_coordination_suggestion())
        
        # 根据运动类型添加特定建议
        if self.sport_type == SportType.BADMINTON:
            suggestions.extend(self._generate_badminton_suggestions(motion_frames))
        elif self.sport_type == SportType.TENNIS:
            suggestions.extend(self._generate_tennis_suggestions(motion_frames))
            
        # 按优先级排序
        suggestions.sort(key=lambda x: x.priority)
        
        return suggestions[:5]  # 最多返回5个建议
    
    def _create_angle_suggestion(self) -> Suggestion:
        """创建角度改进建议"""
        return Suggestion(
            type=FeedbackType.POSTURE,
            priority=1,
            title="改善关节角度控制",
            description="注意保持正确的关节角度，特别是肘部和膝部。建议对着镜子练习标准动作。",
            visual_hint="保持肘部角度在90-150度之间",
            drill_recommendation="每天进行10分钟的关节活动度训练"
        )
    
    def _create_balance_suggestion(self) -> Suggestion:
        """创建平衡改进建议"""
        return Suggestion(
            type=FeedbackType.BALANCE,
            priority=2,
            title="增强身体平衡性",
            description="加强核心力量训练，改善身体稳定性。练习单腿站立等平衡训练。",
            visual_hint="保持重心在双脚之间",
            drill_recommendation="单腿站立训练，每侧30秒×3组"
        )
    
    def _create_fluidity_suggestion(self) -> Suggestion:
        """创建流畅度改进建议"""
        return Suggestion(
            type=FeedbackType.COORDINATION,
            priority=2,
            title="提高动作流畅性",
            description="放松肌肉，避免动作僵硬。可以先慢速练习，逐渐提速。",
            visual_hint="动作要连贯，避免停顿",
            drill_recommendation="慢动作分解练习，每个动作重复20次"
        )
    
    def _create_symmetry_suggestion(self) -> Suggestion:
        """创建对称性改进建议"""
        return Suggestion(
            type=FeedbackType.POSTURE,
            priority=3,
            title="改善左右对称性",
            description="注意左右侧动作的一致性，可以单独训练较弱的一侧。",
            visual_hint="左右动作幅度保持一致",
            drill_recommendation="弱侧单独训练，每天额外15分钟"
        )
    
    def _create_coordination_suggestion(self) -> Suggestion:
        """创建协调性改进建议"""
        return Suggestion(
            type=FeedbackType.COORDINATION,
            priority=1,
            title="提升整体协调性",
            description="加强全身协调训练，注意上下肢配合。可以练习跳绳、游泳等运动。",
            visual_hint="上下肢动作要协调配合",
            drill_recommendation="跳绳训练，每次10分钟"
        )
    
    def _analyze_badminton_strengths(self, motion_frames: List[MotionFrame]) -> List[str]:
        """分析羽毛球特定优点"""
        strengths = []
        
        # 检查挥拍动作
        wrist_angles = []
        for frame in motion_frames:
            for angle in frame.joint_angles:
                if "wrist" in angle.joint_name:
                    wrist_angles.append(angle.angle)
                    
        if wrist_angles and np.mean(wrist_angles) > 120:
            strengths.append("挥拍幅度充分")
            
        # 检查步法
        movement_distance = self._calculate_movement_range(motion_frames)
        if movement_distance > 0.5:  # 假设归一化后的值
            strengths.append("步法移动积极")
            
        return strengths
    
    def _analyze_tennis_strengths(self, motion_frames: List[MotionFrame]) -> List[str]:
        """分析网球特定优点"""
        strengths = []
        # 类似羽毛球的分析逻辑
        return strengths
    
    def _generate_badminton_suggestions(self, motion_frames: List[MotionFrame]) -> List[Suggestion]:
        """生成羽毛球特定建议"""
        suggestions = []
        
        # 检查发球动作
        suggestions.append(Suggestion(
            type=FeedbackType.TIMING,
            priority=2,
            title="优化发球节奏",
            description="发球时注意节奏控制，保持稳定的击球点。",
            visual_hint="击球点在身体前方约30cm处",
            drill_recommendation="定点发球练习，每天50个"
        ))
        
        return suggestions
    
    def _generate_tennis_suggestions(self, motion_frames: List[MotionFrame]) -> List[Suggestion]:
        """生成网球特定建议"""
        suggestions = []
        # 类似羽毛球的建议生成逻辑
        return suggestions
    
    def _calculate_movement_range(self, motion_frames: List[MotionFrame]) -> float:
        """计算移动范围"""
        if len(motion_frames) < 2:
            return 0.0
            
        positions = []
        for frame in motion_frames:
            # 使用髋部中心作为位置参考
            hip_center = (frame.landmarks[23][:2] + frame.landmarks[24][:2]) / 2
            positions.append(hip_center)
            
        positions = np.array(positions)
        movement_range = np.max(positions, axis=0) - np.min(positions, axis=0)
        
        return np.linalg.norm(movement_range)
    
    def _generate_progress_notes(self, current_score: float) -> str:
        """生成进度说明"""
        if not self.feedback_history:
            return "首次训练，继续保持！"
            
        previous_scores = [fb.overall_score for fb in self.feedback_history[-5:]]
        avg_previous = np.mean(previous_scores)
        
        improvement = current_score - avg_previous
        
        if improvement > 5:
            return f"进步明显！相比之前平均水平提升了{improvement:.1f}分"
        elif improvement > 0:
            return f"稳步提升，继续努力！"
        elif improvement > -5:
            return "表现稳定，可以尝试更高难度的训练"
        else:
            return "今天状态略有下降，注意休息和恢复"
    
    def _generate_empty_feedback(self) -> TrainingFeedback:
        """生成空反馈"""
        return TrainingFeedback(
            overall_score=0.0,
            level=FeedbackLevel.NEEDS_IMPROVEMENT,
            strengths=[],
            weaknesses=["未检测到有效动作"],
            suggestions=[
                Suggestion(
                    type=FeedbackType.POSTURE,
                    priority=1,
                    title="调整站位",
                    description="请确保全身在摄像头视野内",
                    visual_hint="站在摄像头前2-3米处"
                )
            ],
            progress_notes="等待动作检测..."
        )
    
    def get_voice_feedback(self, feedback: TrainingFeedback) -> str:
        """生成语音反馈文本"""
        voice_text = f"您的整体得分是{feedback.overall_score:.0f}分，"
        
        if feedback.level == FeedbackLevel.EXCELLENT:
            voice_text += "表现非常出色！"
        elif feedback.level == FeedbackLevel.GOOD:
            voice_text += "表现良好！"
        elif feedback.level == FeedbackLevel.FAIR:
            voice_text += "还有提升空间。"
        else:
            voice_text += "需要加强练习。"
        
        if feedback.strengths:
            voice_text += f"您的优点是：{feedback.strengths[0]}。"
            
        if feedback.suggestions:
            voice_text += f"建议您：{feedback.suggestions[0].description}"
            
        return voice_text
    
    def export_report(self, feedback: TrainingFeedback) -> Dict:
        """导出训练报告"""
        return {
            "score": feedback.overall_score,
            "level": feedback.level.value,
            "strengths": feedback.strengths,
            "weaknesses": feedback.weaknesses,
            "suggestions": [
                {
                    "title": s.title,
                    "description": s.description,
                    "drill": s.drill_recommendation
                }
                for s in feedback.suggestions
            ],
            "progress": feedback.progress_notes
        }