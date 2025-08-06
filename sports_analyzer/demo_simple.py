#!/usr/bin/env python3
"""
简化演示版本 - 仅使用Python标准库
演示系统架构和基本功能
"""

import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import random
import math

class SportsAnalyzerDemo:
    """简化的运动分析器演示"""
    
    def __init__(self):
        self.sessions = {}
        self.analysis_results = []
        
    def create_session(self, sport_type="badminton"):
        """创建训练会话"""
        session_id = f"demo_{int(time.time())}"
        self.sessions[session_id] = {
            "id": session_id,
            "sport_type": sport_type,
            "start_time": time.time(),
            "frame_count": 0,
            "scores": []
        }
        return session_id
    
    def simulate_analysis(self, session_id):
        """模拟动作分析"""
        if session_id not in self.sessions:
            return None
            
        session = self.sessions[session_id]
        
        # 模拟分析结果
        base_score = 75 + random.uniform(-10, 15)
        noise = math.sin(time.time()) * 5
        current_score = max(0, min(100, base_score + noise))
        
        # 模拟常见问题
        issues = []
        if current_score < 60:
            issues.extend(["肘部角度偏差较大", "身体平衡性不足"])
        elif current_score < 80:
            issues.append("动作流畅度需要改善")
        
        # 更新会话数据
        session["frame_count"] += 1
        session["scores"].append(current_score)
        
        return {
            "score": current_score,
            "issues": issues,
            "frame_count": session["frame_count"],
            "duration": time.time() - session["start_time"]
        }
    
    def get_final_feedback(self, session_id):
        """获取最终反馈"""
        if session_id not in self.sessions:
            return None
            
        session = self.sessions[session_id]
        scores = session["scores"]
        
        if not scores:
            return {"error": "No analysis data"}
        
        avg_score = sum(scores) / len(scores)
        
        # 生成反馈
        feedback = {
            "overall_score": avg_score,
            "level": "excellent" if avg_score >= 90 else 
                    "good" if avg_score >= 75 else 
                    "fair" if avg_score >= 60 else "needs_improvement",
            "strengths": ["关节角度控制准确"] if avg_score > 80 else [],
            "weaknesses": ["整体协调性需加强"] if avg_score < 70 else [],
            "suggestions": [
                {
                    "title": "改善关节角度控制",
                    "description": "注意保持正确的关节角度，建议对着镜子练习。",
                    "drill": "每天10分钟关节活动度训练"
                }
            ],
            "progress_notes": f"本次训练平均得分{avg_score:.1f}分，继续保持！"
        }
        
        return feedback

class DemoHTTPHandler(BaseHTTPRequestHandler):
    """HTTP请求处理器"""
    
    analyzer = SportsAnalyzerDemo()
    
    def do_GET(self):
        """处理GET请求"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/':
            self.serve_frontend()
        elif path == '/health':
            self.send_json_response({"status": "healthy"})
        elif path.startswith('/api/v1/session/') and path.endswith('/stats'):
            session_id = path.split('/')[-2]
            stats = self.analyzer.simulate_analysis(session_id)
            self.send_json_response(stats or {"error": "Session not found"})
        else:
            self.send_error(404)
    
    def do_POST(self):
        """处理POST请求"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/api/v1/session/start':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            
            try:
                data = json.loads(post_data) if post_data else {}
                sport_type = data.get('sport_type', 'badminton')
                session_id = self.analyzer.create_session(sport_type)
                
                self.send_json_response({
                    "session_id": session_id,
                    "status": "started",
                    "config": {"sport_type": sport_type}
                })
            except:
                self.send_error(400)
                
        elif path.endswith('/stop'):
            session_id = path.split('/')[-2]
            feedback = self.analyzer.get_final_feedback(session_id)
            self.send_json_response(feedback or {"error": "Session not found"})
            
        else:
            self.send_error(404)
    
    def send_json_response(self, data):
        """发送JSON响应"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
    
    def serve_frontend(self):
        """服务前端页面"""
        html_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI运动训练分析系统 - 演示版</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
        .header { text-align: center; color: #333; margin-bottom: 30px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .score { font-size: 2em; font-weight: bold; text-align: center; margin: 10px 0; }
        .score.excellent { color: #28a745; }
        .score.good { color: #007bff; }
        .score.fair { color: #ffc107; }
        .score.poor { color: #dc3545; }
        button { padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; }
        .btn-primary { background: #007bff; color: white; }
        .btn-success { background: #28a745; color: white; }
        .btn-danger { background: #dc3545; color: white; }
        .stats { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin: 15px 0; }
        .stat { text-align: center; padding: 10px; background: #f8f9fa; border-radius: 5px; }
        .issues { background: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; }
        .demo-note { background: #e7f3ff; border: 1px solid #b3d9ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="demo-note">
            <h3>🎯 演示版说明</h3>
            <p>这是一个使用Python标准库的简化演示版本。实际系统会使用摄像头进行真实的动作分析。</p>
            <p><strong>功能:</strong> 模拟动作分析、实时评分、反馈生成</p>
        </div>
        
        <div class="header">
            <h1>🏸 AI运动训练分析系统</h1>
            <p>基于人工智能的运动训练分析平台</p>
        </div>
        
        <div class="section">
            <h3>训练控制</h3>
            <button id="startBtn" class="btn-primary" onclick="startTraining()">开始训练</button>
            <button id="stopBtn" class="btn-danger" onclick="stopTraining()" disabled>结束训练</button>
            
            <div class="stats">
                <div class="stat">
                    <div><strong id="frameCount">0</strong></div>
                    <div>分析帧数</div>
                </div>
                <div class="stat">
                    <div><strong id="duration">0:00</strong></div>
                    <div>训练时长</div>
                </div>
                <div class="stat">
                    <div><strong id="sessionStatus">待开始</strong></div>
                    <div>状态</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h3>实时评分</h3>
            <div id="scoreDisplay" class="score">--</div>
            
            <div id="issues" class="issues" style="display:none;">
                <h4>检测到的问题:</h4>
                <ul id="issuesList"></ul>
            </div>
        </div>
        
        <div class="section" id="finalFeedback" style="display:none;">
            <h3>训练报告</h3>
            <div id="feedbackContent"></div>
        </div>
    </div>

    <script>
        let sessionId = null;
        let startTime = null;
        let updateInterval = null;

        async function startTraining() {
            try {
                const response = await fetch('/api/v1/session/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sport_type: 'badminton' })
                });
                
                const data = await response.json();
                sessionId = data.session_id;
                startTime = Date.now();
                
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                document.getElementById('sessionStatus').textContent = '进行中';
                document.getElementById('finalFeedback').style.display = 'none';
                
                // 开始实时更新
                updateInterval = setInterval(updateStats, 1000);
                
            } catch (error) {
                alert('启动失败: ' + error.message);
            }
        }

        async function stopTraining() {
            try {
                const response = await fetch(`/api/v1/session/${sessionId}/stop`, {
                    method: 'POST'
                });
                
                const feedback = await response.json();
                displayFeedback(feedback);
                
                // 停止更新
                clearInterval(updateInterval);
                
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
                document.getElementById('sessionStatus').textContent = '已结束';
                
            } catch (error) {
                alert('停止失败: ' + error.message);
            }
        }

        async function updateStats() {
            if (!sessionId) return;
            
            try {
                const response = await fetch(`/api/v1/session/${sessionId}/stats`);
                const stats = await response.json();
                
                if (stats.score !== undefined) {
                    const score = Math.round(stats.score);
                    const scoreElement = document.getElementById('scoreDisplay');
                    scoreElement.textContent = score;
                    
                    // 更新颜色
                    scoreElement.className = 'score ' + 
                        (score >= 90 ? 'excellent' : 
                         score >= 75 ? 'good' : 
                         score >= 60 ? 'fair' : 'poor');
                }
                
                if (stats.frame_count !== undefined) {
                    document.getElementById('frameCount').textContent = stats.frame_count;
                }
                
                if (stats.duration !== undefined) {
                    const duration = Math.floor(stats.duration);
                    const minutes = Math.floor(duration / 60);
                    const seconds = duration % 60;
                    document.getElementById('duration').textContent = 
                        `${minutes}:${seconds.toString().padStart(2, '0')}`;
                }
                
                // 显示问题
                if (stats.issues && stats.issues.length > 0) {
                    const issuesElement = document.getElementById('issues');
                    const issuesListElement = document.getElementById('issuesList');
                    
                    issuesListElement.innerHTML = stats.issues
                        .map(issue => `<li>${issue}</li>`)
                        .join('');
                    issuesElement.style.display = 'block';
                } else {
                    document.getElementById('issues').style.display = 'none';
                }
                
            } catch (error) {
                console.error('更新统计失败:', error);
            }
        }

        function displayFeedback(feedback) {
            const content = document.getElementById('feedbackContent');
            content.innerHTML = `
                <div class="score ${feedback.level}">${Math.round(feedback.overall_score)}</div>
                <p><strong>评级:</strong> ${
                    feedback.level === 'excellent' ? '优秀' :
                    feedback.level === 'good' ? '良好' :
                    feedback.level === 'fair' ? '一般' : '需要改进'
                }</p>
                <p><strong>优点:</strong> ${feedback.strengths.join(', ') || '继续努力'}</p>
                <p><strong>需要改进:</strong> ${feedback.weaknesses.join(', ') || '表现良好'}</p>
                <h4>改进建议:</h4>
                ${feedback.suggestions.map(s => `
                    <div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px;">
                        <strong>${s.title}</strong><br>
                        ${s.description}<br>
                        <em>训练方法: ${s.drill}</em>
                    </div>
                `).join('')}
                <p><em>${feedback.progress_notes}</em></p>
            `;
            
            document.getElementById('finalFeedback').style.display = 'block';
        }
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))
    
    def log_message(self, format, *args):
        """静默日志"""
        pass

def main():
    print("=================================")
    print("  AI运动训练分析系统 - 演示版")
    print("=================================")
    print("这是一个简化演示版本，使用Python标准库实现")
    print("实际版本会集成摄像头和真实的AI分析功能")
    print("")
    print("启动演示服务器...")
    print("访问地址: http://localhost:8080")
    print("按 Ctrl+C 停止服务")
    print("=================================")
    
    server = HTTPServer(('localhost', 8080), DemoHTTPHandler)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n演示服务已停止")
        server.shutdown()

if __name__ == "__main__":
    main()