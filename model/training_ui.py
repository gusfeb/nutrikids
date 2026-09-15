"""
NutriKids — Training Monitor UI
Dashboard untuk memantau proses training secara real-time.
Berjalan di http://127.0.0.1:5002
"""

from flask import Flask, render_template_string, jsonify
import os
import json
import subprocess

app = Flask(__name__)

STATUS_FILE = os.path.join(os.path.dirname(__file__), 'training_status.json')
TRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), 'train_model.py')

# Menyimpan referensi ke proses yang sedang berjalan
training_process = None

# ============================================================
# TEMPLATE HTML
# ============================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>NutriKids // Cyber-Training Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --cp-yellow: #FCEE0A;
            --cp-cyan: #00F0FF;
            --cp-red: #FF003C;
            --cp-dark: #050505;
            --cp-gray: #1a1a1a;
            --cp-light: #d8d8d8;
        }
        body { 
            font-family: 'Rajdhani', sans-serif; 
            background: var(--cp-dark); 
            background-image: linear-gradient(0deg, transparent 24%, rgba(0, 240, 255, 0.05) 25%, rgba(0, 240, 255, 0.05) 26%, transparent 27%, transparent 74%, rgba(0, 240, 255, 0.05) 75%, rgba(0, 240, 255, 0.05) 76%, transparent 77%, transparent), linear-gradient(90deg, transparent 24%, rgba(0, 240, 255, 0.05) 25%, rgba(0, 240, 255, 0.05) 26%, transparent 27%, transparent 74%, rgba(0, 240, 255, 0.05) 75%, rgba(0, 240, 255, 0.05) 76%, transparent 77%, transparent);
            background-size: 30px 30px;
            color: var(--cp-light); 
            margin: 0; 
            padding: 20px; 
            text-transform: uppercase;
        }
        .container { max-width: 1100px; margin: auto; }
        .header { 
            display: flex; justify-content: space-between; align-items: flex-start; 
            background: var(--cp-gray); 
            padding: 20px 30px; 
            border-left: 5px solid var(--cp-yellow);
            border-right: 5px solid var(--cp-cyan);
            clip-path: polygon(0 0, 100% 0, 100% calc(100% - 15px), calc(100% - 15px) 100%, 0 100%);
            margin-bottom: 25px; 
            position: relative;
        }
        .header::after {
            content: "SYS_CTRL";
            position: absolute;
            right: 20px;
            bottom: -5px;
            font-size: 0.6rem;
            color: var(--cp-cyan);
            letter-spacing: 2px;
        }
        h1 { 
            font-family: 'Orbitron', sans-serif; 
            margin: 0; 
            color: var(--cp-yellow); 
            text-shadow: 2px 2px 0px var(--cp-red);
            font-size: 2.2rem;
            letter-spacing: 1px;
        }
        p { margin: 5px 0 0 0; color: var(--cp-cyan); font-weight: 600; letter-spacing: 1px;}
        .status-badge { 
            padding: 6px 15px; 
            font-weight: 700; 
            font-size: 1rem; 
            border: 1px solid currentColor;
            background: rgba(0,0,0,0.5);
            font-family: 'Orbitron', sans-serif;
            clip-path: polygon(10px 0, 100% 0, 100% calc(100% - 10px), calc(100% - 10px) 100%, 0 100%, 0 10px);
        }
        .status-idle { color: var(--cp-light); }
        .status-running { color: var(--cp-yellow); text-shadow: 0 0 5px var(--cp-yellow); box-shadow: inset 0 0 10px rgba(252, 238, 10, 0.2); }
        .status-done { color: var(--cp-cyan); text-shadow: 0 0 5px var(--cp-cyan); }
        
        .btn { 
            padding: 12px 25px; 
            border: none; 
            font-family: 'Orbitron', sans-serif;
            font-weight: 900; 
            cursor: pointer; 
            transition: all 0.1s; 
            font-size: 1.1rem; 
            clip-path: polygon(15px 0, 100% 0, 100% calc(100% - 15px), calc(100% - 15px) 100%, 0 100%, 0 15px);
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .btn-start { 
            background: var(--cp-red); 
            color: white; 
            border-bottom: 3px solid #b3002a;
        }
        .btn-start:hover:not(:disabled) { 
            background: var(--cp-cyan); 
            color: #000;
            border-bottom: 3px solid #0096a3;
            box-shadow: 0 0 15px var(--cp-cyan);
        }
        .btn-start:disabled { 
            background: #333; color: #666; cursor: not-allowed; border-bottom: none;
        }
        
        .metrics-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px; margin-bottom: 25px; }
        .metric-card { 
            background: var(--cp-gray); 
            padding: 15px; 
            border-top: 3px solid var(--cp-yellow);
            position: relative;
            text-align: center; 
            clip-path: polygon(0 0, 100% 0, 100% calc(100% - 15px), calc(100% - 15px) 100%, 0 100%);
        }
        .metric-card::before {
            content: '';
            position: absolute;
            left: 0; bottom: 0;
            width: 30px; height: 3px;
            background: var(--cp-cyan);
        }
        .metric-title { font-size: 0.9rem; color: #888; font-weight: 700; letter-spacing: 2px; margin-bottom: 8px; }
        .metric-value { font-size: 2rem; font-weight: 700; color: var(--cp-cyan); text-shadow: 0 0 10px rgba(0, 240, 255, 0.4); font-family: 'Orbitron', sans-serif;}
        
        .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .chart-container { 
            background: var(--cp-gray); 
            padding: 20px; 
            border: 1px solid #333;
            position: relative;
        }
        .chart-container::before, .chart-container::after {
            content: '';
            position: absolute;
            width: 15px; height: 15px;
            border: 2px solid var(--cp-yellow);
        }
        .chart-container::before { top: -2px; left: -2px; border-right: none; border-bottom: none; }
        .chart-container::after { bottom: -2px; right: -2px; border-left: none; border-top: none; }

        /* CYBER ANIMATIONS */
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
        .blink { animation: blink 1s infinite; }
        
        @keyframes glitch {
            0% { transform: translate(0) }
            20% { transform: translate(-2px, 1px) }
            40% { transform: translate(-1px, -1px) }
            60% { transform: translate(2px, 1px) }
            80% { transform: translate(1px, -1px) }
            100% { transform: translate(0) }
        }
        .glitch-anim { animation: glitch 0.2s infinite; display: inline-block; color: var(--cp-red); }

        .progress-container {
            width: 100%; height: 3px; background: #333; margin-top: 15px;
            position: relative; overflow: hidden; display: none;
        }
        .progress-bar {
            height: 100%; width: 30%;
            background: var(--cp-cyan); box-shadow: 0 0 15px var(--cp-cyan);
            animation: scanline 1.5s infinite linear;
        }
        @keyframes scanline {
            0% { transform: translateX(-100%); width: 10%; }
            50% { width: 30%; }
            100% { transform: translateX(400%); width: 10%; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>//_TRAINING_MATRIX</h1>
                <p>PHASE: <span id="current-phase" style="color: var(--cp-light);">AWAITING_INPUT</span></p>
                <div id="cyber-loader" style="display: none; margin-top: 10px; font-family: 'Orbitron', sans-serif; color: var(--cp-yellow); font-size: 0.95rem; letter-spacing: 2px;">
                    <span class="glitch-anim">SYS_PROCESS</span> // INGESTING_NEURAL_DATA... <span class="blink">_</span> [<span id="loading-chars" style="color: var(--cp-cyan);">0XF8</span>]
                    <div class="progress-container" id="progress-container">
                        <div class="progress-bar"></div>
                    </div>
                </div>
            </div>
            <div style="display: flex; gap: 15px; align-items: center; align-self: center;">
                <span id="status-badge" class="status-badge status-idle">IDLE_STATE</span>
                <button id="start-btn" class="btn btn-start" onclick="startTraining()">INIT_TRAINING</button>
            </div>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">TIME ELAPSED</div>
                <div class="metric-value" id="val-time">00:00</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">CYCLES [EPOCH]</div>
                <div class="metric-value" id="val-epoch">00</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">TRAIN ACCURACY</div>
                <div class="metric-value" id="val-acc">--%</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">VAL ACCURACY</div>
                <div class="metric-value" id="val-val-acc">--%</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">LOSS DELTA</div>
                <div class="metric-value" id="val-loss">-.--</div>
            </div>
        </div>

        <div class="charts">
            <div class="chart-container">
                <canvas id="accChart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="lossChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        // Matrix Char Animation
        const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%&*";
        setInterval(() => {
            if(document.getElementById('cyber-loader').style.display === 'block') {
                let str = '';
                for(let i=0; i<4; i++) str += chars.charAt(Math.floor(Math.random() * chars.length));
                document.getElementById('loading-chars').innerText = str;
            }
        }, 80);

        // Setup Chart.js with Cyberpunk Theme
        Chart.defaults.color = '#888';
        Chart.defaults.font.family = "'Rajdhani', sans-serif";
        const gridColor = '#333';
        
        const accCtx = document.getElementById('accChart').getContext('2d');
        const lossCtx = document.getElementById('lossChart').getContext('2d');
        
        const commonOptions = {
            responsive: true,
            animation: { duration: 200 },
            scales: { 
                x: { grid: { color: gridColor, drawBorder: false } },
                y: { grid: { color: gridColor, drawBorder: false } }
            },
            plugins: {
                legend: { labels: { color: '#d8d8d8', font: { weight: 'bold' } } }
            }
        };

        const accChart = new Chart(accCtx, {
            type: 'line',
            data: { labels: [], datasets: [
                { label: 'TRAIN_ACC', borderColor: '#00F0FF', backgroundColor: 'rgba(0, 240, 255, 0.1)', data: [], tension: 0, borderWidth: 2, pointBackgroundColor: '#00F0FF', pointRadius: 2, fill: true },
                { label: 'VAL_ACC', borderColor: '#FCEE0A', data: [], tension: 0, borderWidth: 2, pointBackgroundColor: '#FCEE0A', pointRadius: 2, borderDash: [5, 5] }
            ]},
            options: { ...commonOptions, plugins: { ...commonOptions.plugins, title: { display: true, text: 'ACCURACY_MATRIX', color: '#00F0FF', font: { family: "'Orbitron', sans-serif", size: 14 } } } }
        });

        const lossChart = new Chart(lossCtx, {
            type: 'line',
            data: { labels: [], datasets: [
                { label: 'TRAIN_LOSS', borderColor: '#FF003C', backgroundColor: 'rgba(255, 0, 60, 0.1)', data: [], tension: 0, borderWidth: 2, pointBackgroundColor: '#FF003C', pointRadius: 2, fill: true },
                { label: 'VAL_LOSS', borderColor: '#FCEE0A', data: [], tension: 0, borderWidth: 2, pointBackgroundColor: '#FCEE0A', pointRadius: 2, borderDash: [5, 5] }
            ]},
            options: { ...commonOptions, plugins: { ...commonOptions.plugins, title: { display: true, text: 'LOSS_DEGRADATION', color: '#FF003C', font: { family: "'Orbitron', sans-serif", size: 14 } } } }
        });

        let currentEpoch = 0;
        let startTime = null;
        let timerInterval = null;

        function updateTimer() {
            if (!startTime) return;
            const diff = Math.floor((Date.now() - startTime) / 1000);
            const m = Math.floor(diff / 60).toString().padStart(2, '0');
            const s = (diff % 60).toString().padStart(2, '0');
            document.getElementById('val-time').innerText = `${m}:${s}`;
        }

        function updateDashboard() {
            fetch('/status')
                .then(res => res.json())
                .then(data => {
                    if (data.status === 'idle') return;
                    
                    document.getElementById('current-phase').innerText = data.phase.toUpperCase();
                    
                    const badge = document.getElementById('status-badge');
                    badge.innerText = `STATUS: ${data.status.toUpperCase()}`;
                    
                    if (data.status === 'Training' || data.status === 'Menyiapkan...') {
                        if (!startTime && data.status === 'Training') startTime = Date.now();
                        if (!timerInterval && data.status === 'Training') timerInterval = setInterval(updateTimer, 1000);
                        
                        if (data.status === 'Training') {
                            badge.className = 'status-badge status-running';
                            document.getElementById('start-btn').disabled = true;
                            document.getElementById('start-btn').innerText = 'TRAINING...';
                            document.getElementById('cyber-loader').style.display = 'block';
                            document.getElementById('progress-container').style.display = 'block';
                        }
                    } else if (data.status === 'Selesai') {
                        badge.className = 'status-badge status-done';
                        document.getElementById('start-btn').disabled = false;
                        document.getElementById('start-btn').innerText = 'REBOOT_TRAINING';
                        document.getElementById('cyber-loader').style.display = 'none';
                        document.getElementById('progress-container').style.display = 'none';
                        
                        if (timerInterval) {
                            clearInterval(timerInterval);
                            timerInterval = null;
                        }
                    }

                    if (data.epoch !== 'Selesai' && data.epoch > currentEpoch) {
                        currentEpoch = data.epoch;
                        document.getElementById('val-epoch').innerText = data.epoch.toString().padStart(2, '0');
                        
                        if (data.logs.binary_accuracy) {
                            const acc = (data.logs.binary_accuracy * 100).toFixed(1) + '%';
                            const valAcc = (data.logs.val_binary_accuracy * 100).toFixed(1) + '%';
                            document.getElementById('val-acc').innerText = acc;
                            document.getElementById('val-val-acc').innerText = valAcc;
                            document.getElementById('val-loss').innerText = data.logs.loss.toFixed(4);
                            
                            // Update charts
                            const label = `E${data.epoch}`;
                            accChart.data.labels.push(label);
                            accChart.data.datasets[0].data.push(data.logs.binary_accuracy);
                            accChart.data.datasets[1].data.push(data.logs.val_binary_accuracy);
                            accChart.update();
                            
                            lossChart.data.labels.push(label);
                            lossChart.data.datasets[0].data.push(data.logs.loss);
                            lossChart.data.datasets[1].data.push(data.logs.val_loss);
                            lossChart.update();
                        }
                    }
                })
                .catch(err => console.log('SYS_ERROR:', err));
        }

        function startTraining() {
            if (!confirm('WARNING: OVERWRITE EXISTING NEURAL NETWORK? INIT SEQUENCE?')) return;
            
            document.getElementById('start-btn').disabled = true;
            document.getElementById('status-badge').innerText = 'STATUS: PREP...';
            document.getElementById('status-badge').className = 'status-badge status-running';
            document.getElementById('cyber-loader').style.display = 'block';
            document.getElementById('progress-container').style.display = 'block';
            
            // Setup Timer
            startTime = Date.now();
            if (timerInterval) clearInterval(timerInterval);
            timerInterval = setInterval(updateTimer, 1000);
            document.getElementById('val-time').innerText = '00:00';
            
            // Reset charts
            accChart.data.labels = []; accChart.data.datasets.forEach(d => d.data = []); accChart.update();
            lossChart.data.labels = []; lossChart.data.datasets.forEach(d => d.data = []); lossChart.update();
            currentEpoch = 0;
            
            fetch('/start', { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    console.log('INIT:', data);
                });
        }

        // Poll status every 2 seconds
        setInterval(updateDashboard, 2000);
        updateDashboard();
    </script>
</body>
</html>
"""

# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def status():
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r') as f:
                return jsonify(json.load(f))
        except:
            return jsonify({'status': 'reading...'})
    return jsonify({'status': 'idle'})

@app.route('/start', methods=['POST'])
def start():
    global training_process
    
    # Hanya jalankan jika tidak ada proses yang sedang berjalan
    if training_process is None or training_process.poll() is not None:
        # Reset file status
        with open(STATUS_FILE, 'w') as f:
            json.dump({'status': 'Menyiapkan...', 'phase': 'Inisialisasi', 'epoch': 0, 'logs': {}}, f)
            
        # Jalankan script di background
        # Kita panggil dengan sys.executable agar menggunakan venv yang sedang aktif
        import sys
        training_process = subprocess.Popen([sys.executable, TRAIN_SCRIPT])
        return jsonify({'message': 'Training started'})
    else:
        return jsonify({'message': 'Training is already running!'}), 400

if __name__ == '__main__':
    print("==================================================")
    print("NutriKids Training Monitor Aktif")
    print("Buka browser dan akses: http://127.0.0.1:5002")
    print("==================================================")
    # File status tidak lagi dihapus di sini agar aman saat auto-reload
    app.run(debug=True, port=5002)
