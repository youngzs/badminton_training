# AI运动训练分析系统

## 系统概述

这是一个基于人工智能的运动训练分析系统，通过计算机视觉技术实时分析运动员的动作，提供专业的动作纠正建议和训练反馈。系统支持多种运动类型，包括羽毛球、网球、篮球、高尔夫、瑜伽和跑步。

## 核心功能

- **实时动作捕捉与分析**：使用MediaPipe进行实时姿势检测
- **智能评分系统**：基于多维度指标的动作质量评估
- **个性化反馈**：根据动作问题生成针对性的改进建议
- **训练记录**：记录和分析训练数据，跟踪进步
- **多运动支持**：适配不同运动的特定动作分析
- **视频分析**：支持上传视频文件进行离线分析

## 技术架构

### 后端技术栈
- **Python 3.12+**：主要开发语言
- **FastAPI**：高性能Web框架
- **MediaPipe**：Google的姿势检测库
- **OpenCV**：视频处理
- **NumPy/SciPy**：数值计算
- **WebSocket**：实时通信

### 前端技术栈
- **HTML5/CSS3/JavaScript**：Web界面
- **WebRTC**：摄像头访问
- **Canvas**：数据可视化

## 项目结构

```
sports_analyzer/
├── core/                   # 核心功能模块
│   ├── motion_analyzer.py  # 动作分析引擎
│   ├── feedback_system.py  # 智能反馈系统
│   └── video_processor.py  # 视频处理模块
├── api/                    # API服务
│   └── main.py            # FastAPI应用
├── frontend/              # 前端界面
│   └── index.html        # Web界面
├── models/               # AI模型（预留）
├── utils/                # 工具函数
├── tests/                # 测试用例
├── config/               # 配置文件
├── requirements.txt      # 依赖列表
├── run.sh               # 启动脚本
└── README.md            # 项目说明

```

## 快速开始

### 1. 环境要求

- Python 3.8+
- 摄像头（用于实时分析）
- 现代浏览器（Chrome/Firefox/Safari）

### 2. 安装步骤

```bash
# 克隆项目
cd sports_analyzer

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 启动系统

```bash
# 方式1：使用启动脚本
chmod +x run.sh
./run.sh

# 方式2：手动启动
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 访问系统

- **API文档**：http://localhost:8000/docs
- **前端界面**：在浏览器中打开 `frontend/index.html`

## 使用指南

### 实时分析模式

1. 打开前端界面
2. 允许浏览器访问摄像头
3. 选择运动类型
4. 点击"开始训练"
5. 进行运动动作
6. 查看实时反馈
7. 点击"结束"获取详细分析报告

### 视频上传分析

1. 点击"选择视频文件"
2. 选择要分析的视频（支持mp4/avi/mov格式）
3. 等待分析完成
4. 查看分析结果和建议

## API接口说明

### 主要端点

| 端点 | 方法 | 描述 |
|-----|------|-----|
| `/api/v1/session/start` | POST | 开始训练会话 |
| `/api/v1/session/{id}/stop` | POST | 结束训练会话 |
| `/api/v1/session/{id}/stats` | GET | 获取会话统计 |
| `/api/v1/analysis/upload` | POST | 上传视频分析 |
| `/api/v1/analysis/frame` | POST | 分析单帧图像 |
| `/ws/{session_id}` | WebSocket | 实时数据流 |

### 示例代码

```python
import requests

# 开始训练会话
response = requests.post('http://localhost:8000/api/v1/session/start', 
    json={
        "sport_type": "badminton",
        "video_source": "webcam"
    })
session_id = response.json()['session_id']

# 获取统计信息
stats = requests.get(f'http://localhost:8000/api/v1/session/{session_id}/stats')
print(stats.json())

# 结束会话并获取反馈
feedback = requests.post(f'http://localhost:8000/api/v1/session/{session_id}/stop')
print(feedback.json())
```

## 评分算法

系统采用多维度评分机制：

1. **关节角度准确性 (40%)**
   - 计算各关节角度与标准值的偏差
   - 根据运动类型设定最优角度范围

2. **动作流畅度 (30%)**
   - 分析速度变化率
   - 评估动作连贯性

3. **身体平衡性 (20%)**
   - 重心稳定度分析
   - 左右对称性检测

4. **时序准确性 (10%)**
   - 动作节奏评估
   - 反应时间分析

## 自定义配置

### 运动参数调整

编辑 `core/motion_analyzer.py` 中的 `_get_optimal_angles()` 方法来调整不同运动的标准角度：

```python
def _get_optimal_angles(self):
    if self.sport_type == SportType.BADMINTON:
        return {
            'left_elbow': (90, 150),  # 调整角度范围
            'right_elbow': (90, 150),
            # ...
        }
```

### 反馈系统定制

在 `core/feedback_system.py` 中自定义反馈生成逻辑：

```python
def _generate_suggestions(self, motion_frames, weaknesses):
    # 添加自定义建议生成逻辑
    pass
```

## 性能优化

- **GPU加速**：如果有NVIDIA GPU，安装CUDA版本的TensorFlow/PyTorch
- **降低分辨率**：在 `VideoConfig` 中调整分辨率以提高处理速度
- **帧率控制**：根据硬件性能调整FPS设置

## 常见问题

### Q: 摄像头无法访问？
A: 确保浏览器有摄像头权限，使用HTTPS或localhost访问

### Q: 分析精度不高？
A: 
- 确保良好的照明条件
- 摄像头与运动员距离2-3米
- 全身在画面内
- 避免复杂背景

### Q: 系统运行缓慢？
A: 
- 降低视频分辨率
- 关闭其他占用GPU的程序
- 使用更高配置的硬件

## 开发计划

- [ ] 添加更多运动类型支持
- [ ] 集成深度学习模型优化精度
- [ ] 开发移动端APP
- [ ] 添加多人同时分析
- [ ] 实现云端部署
- [ ] 添加教练端管理系统
- [ ] 支持3D动作重建
- [ ] 集成语音指导功能

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 许可证

MIT License

## 联系方式

如有问题或建议，请创建Issue或联系项目维护者。

## 致谢

- MediaPipe团队提供的优秀姿势检测库
- FastAPI框架的高性能支持
- 开源社区的贡献和支持