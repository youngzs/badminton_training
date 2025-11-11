<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI运动训练分析系统</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        header {
            background: rgba(255, 255, 255, 0.95);
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .header-content {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        h1 {
            color: #333;
            font-size: 1.8rem;
        }
        
        .sport-selector {
            padding: 0.5rem 1rem;
            border-radius: 5px;
            border: 1px solid #ddd;
            font-size: 1rem;
        }
        
        main {
            flex: 1;
            padding: 2rem;
            max-width: 1200px;
            margin: 0 auto;
            width: 100%;
        }
        
        .container {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 2rem;
            height: 100%;
        }
        
        .video-section {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        
        .video-container {
            position: relative;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            aspect-ratio: 16/9;
            margin-bottom: 1rem;
        }
        
        #videoElement {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .video-overlay {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-size: 14px;
        }
        
        .controls {
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
        }
        
        button {
            flex: 1;
            padding: 0.8rem 1.5rem;
            border: none;
            border-radius: 5px;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn-primary {
            background: #667eea;
            color: white;
        }
        
        .btn-primary:hover {
            background: #5a67d8;
        }
        
        .btn-secondary {
            background: #48bb78;
            color: white;
        }
        
        .btn-secondary:hover {
            background: #38a169;
        }
        
        .btn-danger {
            background: #f56565;
            color: white;
        }
        
        .btn-danger:hover {
            background: #e53e3e;
        }
        
        .feedback-section {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .card {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        
        .card h2 {
            color: #333;
            margin-bottom: 1rem;
            font-size: 1.3rem;
        }
        
        .score-display {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            margin: 1rem 0;
        }
        
        .score-excellent { color: #48bb78; }
        .score-good { color: #4299e1; }
        .score-fair { color: #ed8936; }
        .score-poor { color: #f56565; }
        
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .stat-item {
            text-align: center;
            padding: 0.5rem;
            background: #f7fafc;
            border-radius: 5px;
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: #718096;
        }
        
        .feedback-list {
            list-style: none;
            padding: 0;
        }
        
        .feedback-item {
            padding: 0.5rem;
            margin: 0.5rem 0;
            background: #f7fafc;
            border-left: 3px solid #667eea;
            border-radius: 3px;
        }
        
        .upload-section {
            margin-top: 2rem;
            padding: 2rem;
            border: 2px dashed #cbd5e0;
            border-radius: 10px;
            text-align: center;
            background: #f7fafc;
        }
        
        .upload-input {
            display: none;
        }
        
        .upload-label {
            display: inline-block;
            padding: 0.8rem 2rem;
            background: #667eea;
            color: white;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.3s;
        }
        
        .upload-label:hover {
            background: #5a67d8;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 2rem;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .progress-chart {
            height: 200px;
            background: #f7fafc;
            border-radius: 5px;
            padding: 1rem;
            margin-top: 1rem;
        }
        
        @media (max-width: 768px) {
            .container {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <header>
        <div class="header-content">
            <h1>🏸 AI运动训练分析系统</h1>
            <select class="sport-selector" id="sportSelector">
                <option value="badminton">羽毛球</option>
                <option value="tennis">网球</option>
                <option value="basketball">篮球</option>
                <option value="golf">高尔夫</option>
                <option value="yoga">瑜伽</option>
                <option value="running">跑步</option>
            </select>
        </div>
    </header>
    
    <main>
        <div class="container">
            <div class="video-section">
                <div class="video-container">
                    <video id="videoElement" autoplay></video>
                    <div class="video-overlay">
                        <div id="fpsDisplay">FPS: 0</div>
                        <div id="statusDisplay">准备就绪</div>
                    </div>
                </div>
                
                <div class="controls">
                    <button id="startBtn" class="btn-primary">开始训练</button>
                    <button id="pauseBtn" class="btn-secondary" disabled>暂停</button>
                    <button id="stopBtn" class="btn-danger" disabled>结束</button>
                </div>
                
                <div class="upload-section">
                    <p>或者上传视频文件进行分析</p>
                    <input type="file" id="uploadInput" class="upload-input" accept="video/*">
                    <label for="uploadInput" class="upload-label">选择视频文件</label>
                </div>
                
                <div class="loading" id="loadingIndicator">
                    <div class="spinner"></div>
                    <p>正在分析中...</p>
                </div>
            </div>
            
            <div class="feedback-section">
                <div class="card">
                    <h2>实时评分</h2>
                    <div id="scoreDisplay" class="score-display score-good">--</div>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-value" id="frameCount">0</div>
                            <div class="stat-label">分析帧数</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="duration">0:00</div>
                            <div class="stat-label">训练时长</div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>动作问题</h2>
                    <ul class="feedback-list" id="issuesList">
                        <li class="feedback-item">等待分析开始...</li>
                    </ul>
                </div>
                
                <div class="card">
                    <h2>改进建议</h2>
                    <ul class="feedback-list" id="suggestionsList">
                        <li class="feedback-item">完成训练后查看详细建议</li>
                    </ul>
                </div>
                
                <div class="card">
                    <h2>训练进度</h2>
                    <div class="progress-chart" id="progressChart">
                        <canvas id="chartCanvas"></canvas>
                    </div>
                </div>
            </div>
        </div>
    </main>
    
    <script>
        // API配置
        const API_BASE = 'http://localhost:8000';
        let sessionId = null;
        let ws = null;
        let startTime = null;
        let animationFrame = null;
        
        // DOM元素
        const videoElement = document.getElementById('videoElement');
        const startBtn = document.getElementById('startBtn');
        const pauseBtn = document.getElementById('pauseBtn');
        const stopBtn = document.getElementById('stopBtn');
        const uploadInput = document.getElementById('uploadInput');
        const sportSelector = document.getElementById('sportSelector');
        const scoreDisplay = document.getElementById('scoreDisplay');
        const frameCount = document.getElementById('frameCount');
        const duration = document.getElementById('duration');
        const issuesList = document.getElementById('issuesList');
        const suggestionsList = document.getElementById('suggestionsList');
        const fpsDisplay = document.getElementById('fpsDisplay');
        const statusDisplay = document.getElementById('statusDisplay');
        const loadingIndicator = document.getElementById('loadingIndicator');
        
        // 初始化摄像头
        async function initCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: 1280,
                        height: 720,
                        facingMode: 'user'
                    }
                });
                videoElement.srcObject = stream;
                statusDisplay.textContent = '摄像头已就绪';
            } catch (error) {
                console.error('Camera initialization failed:', error);
                statusDisplay.textContent = '摄像头初始化失败';
            }
        }
        
        // 开始训练会话
        async function startSession() {
            try {
                const response = await fetch(`${API_BASE}/api/v1/session/start`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        sport_type: sportSelector.value,
                        video_source: 'webcam',
                        enable_recording: false
                    })
                });
                
                const data = await response.json();
                sessionId = data.session_id;
                
                // 连接WebSocket
                connectWebSocket();
                
                // 更新UI
                startBtn.disabled = true;
                pauseBtn.disabled = false;
                stopBtn.disabled = false;
                statusDisplay.textContent = '训练进行中';
                startTime = Date.now();
                
                // 开始更新时间
                updateDuration();
                
            } catch (error) {
                console.error('Failed to start session:', error);
                alert('启动训练失败');
            }
        }
        
        // 连接WebSocket
        function connectWebSocket() {
            ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'stats') {
                    updateStats(data.data);
                }
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
            ws.onclose = () => {
                console.log('WebSocket connection closed');
            };
        }
        
        // 更新统计信息
        function updateStats(stats) {
            if (stats.actual_fps) {
                fpsDisplay.textContent = `FPS: ${stats.actual_fps.toFixed(1)}`;
            }
            
            if (stats.frame_count) {
                frameCount.textContent = stats.frame_count;
            }
            
            if (stats.current_score !== undefined) {
                const score = Math.round(stats.current_score * 100);
                scoreDisplay.textContent = score;
                
                // 更新颜色
                scoreDisplay.className = 'score-display';
                if (score >= 90) {
                    scoreDisplay.classList.add('score-excellent');
                } else if (score >= 75) {
                    scoreDisplay.classList.add('score-good');
                } else if (score >= 60) {
                    scoreDisplay.classList.add('score-fair');
                } else {
                    scoreDisplay.classList.add('score-poor');
                }
            }
        }
        
        // 更新训练时长
        function updateDuration() {
            if (startTime) {
                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                const minutes = Math.floor(elapsed / 60);
                const seconds = elapsed % 60;
                duration.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
                
                animationFrame = requestAnimationFrame(updateDuration);
            }
        }
        
        // 停止训练会话
        async function stopSession() {
            try {
                loadingIndicator.style.display = 'block';
                
                const response = await fetch(`${API_BASE}/api/v1/session/${sessionId}/stop`, {
                    method: 'POST'
                });
                
                const feedback = await response.json();
                
                // 显示反馈
                displayFeedback(feedback);
                
                // 关闭WebSocket
                if (ws) {
                    ws.close();
                }
                
                // 重置UI
                startBtn.disabled = false;
                pauseBtn.disabled = true;
                stopBtn.disabled = true;
                statusDisplay.textContent = '训练结束';
                
                // 停止时间更新
                cancelAnimationFrame(animationFrame);
                startTime = null;
                sessionId = null;
                
            } catch (error) {
                console.error('Failed to stop session:', error);
                alert('停止训练失败');
            } finally {
                loadingIndicator.style.display = 'none';
            }
        }
        
        // 显示反馈
        function displayFeedback(feedback) {
            // 显示最终得分
            scoreDisplay.textContent = Math.round(feedback.overall_score);
            
            // 显示问题
            if (feedback.weaknesses && feedback.weaknesses.length > 0) {
                issuesList.innerHTML = feedback.weaknesses
                    .map(issue => `<li class="feedback-item">❗ ${issue}</li>`)
                    .join('');
            }
            
            // 显示建议
            if (feedback.suggestions && feedback.suggestions.length > 0) {
                suggestionsList.innerHTML = feedback.suggestions
                    .map(suggestion => `
                        <li class="feedback-item">
                            <strong>${suggestion.title}</strong><br>
                            ${suggestion.description}
                            ${suggestion.drill ? `<br><em>训练方法: ${suggestion.drill}</em>` : ''}
                        </li>
                    `)
                    .join('');
            }
            
            // 显示优点
            if (feedback.strengths && feedback.strengths.length > 0) {
                const strengthsHtml = feedback.strengths
                    .map(strength => `<li class="feedback-item">✅ ${strength}</li>`)
                    .join('');
                    
                issuesList.innerHTML = `
                    <h3>优点</h3>
                    ${strengthsHtml}
                    <h3>需要改进</h3>
                    ${issuesList.innerHTML}
                `;
            }
        }
        
        // 上传视频分析
        async function uploadVideo(file) {
            try {
                loadingIndicator.style.display = 'block';
                
                const formData = new FormData();
                formData.append('file', file);
                
                const response = await fetch(`${API_BASE}/api/v1/analysis/upload`, {
                    method: 'POST',
                    body: formData
                });
                
                const feedback = await response.json();
                displayFeedback(feedback);
                
            } catch (error) {
                console.error('Failed to upload video:', error);
                alert('视频上传分析失败');
            } finally {
                loadingIndicator.style.display = 'none';
            }
        }
        
        // 事件监听
        startBtn.addEventListener('click', startSession);
        stopBtn.addEventListener('click', stopSession);
        
        pauseBtn.addEventListener('click', () => {
            // 暂停功能实现
            alert('暂停功能开发中...');
        });
        
        uploadInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file) {
                uploadVideo(file);
            }
        });
        
        // 初始化
        initCamera();
    </script>
</body>
</html>