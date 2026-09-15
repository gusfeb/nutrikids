/**
 * NutriKids — Main JavaScript
 * Handles: Image upload, drag-and-drop, preview, API calls, result display,
 *          and detection overlay visualization
 */

// ============================================================
// ELEMENTS
// ============================================================
const uploadZone = document.getElementById('upload-zone');
const fotoInput = document.getElementById('foto-input');
const uploadPlaceholder = document.getElementById('upload-placeholder');
const previewArea = document.getElementById('preview-area');
const previewImg = document.getElementById('preview-img');
const analyzeBtn = document.getElementById('analyze-btn');
const loadingOverlay = document.getElementById('loading-overlay');
const resultContainer = document.getElementById('result-container');
const resultScore = document.getElementById('result-score');
const resultSummary = document.getElementById('result-summary');
const resultGrid = document.getElementById('result-grid');
const detectionCanvas = document.getElementById('detection-canvas');
const detectionLegend = document.getElementById('detection-legend');
const legendItems = document.getElementById('legend-items');

let selectedFile = null;

// ============================================================
// DETECTION BOX CONFIG — warna & posisi tiap komponen
// ============================================================
const DETECTION_CONFIG = {
    makanan_pokok: {
        color: '#AB47BC',       // ungu
        label: 'Karbohidrat',
        icon: '🍚',
        // Posisi relatif (0-1) di dalam gambar: [x, y, width, height]
        boxes: [
            { x: 0.55, y: 0.05, w: 0.38, h: 0.42 }
        ]
    },
    lauk_pauk: {
        color: '#00E5FF',       // cyan
        label: 'Protein',
        icon: '🍗',
        boxes: [
            { x: 0.25, y: 0.48, w: 0.45, h: 0.48 }
        ]
    },
    sayur: {
        color: '#FF4081',       // pink
        label: 'Sayuran',
        icon: '🥬',
        boxes: [
            { x: 0.65, y: 0.50, w: 0.32, h: 0.45 }
        ]
    },
    buah: {
        color: '#FFAB00',       // kuning/oranye
        label: 'Buah-buahan',
        icon: '🍎',
        boxes: [
            { x: 0.02, y: 0.50, w: 0.28, h: 0.45 }
        ]
    },
    susu: {
        color: '#FF6B35',       // oranye/coral
        label: 'Susu',
        icon: '🥛',
        boxes: [
            { x: 0.20, y: 0.05, w: 0.35, h: 0.22 }
        ]
    }
};

// ============================================================
// DRAG & DROP
// ============================================================
if (uploadZone) {
    uploadZone.addEventListener('click', (e) => {
        if (e.target.closest('button')) return;
        fotoInput.click();
    });

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleFile(file);
        }
    });
}

// ============================================================
// FILE INPUT CHANGE
// ============================================================
if (fotoInput) {
    fotoInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleFile(file);
    });
}

// ============================================================
// HANDLE FILE — Preview
// ============================================================
function handleFile(file) {
    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        uploadPlaceholder.style.display = 'none';
        previewArea.style.display = 'block';
        uploadZone.classList.add('has-preview');
        analyzeBtn.style.display = 'inline-flex';
        resultContainer.classList.remove('show');
        // Clear previous detection
        clearDetectionOverlay();
    };
    reader.readAsDataURL(file);
}

// ============================================================
// ANALYZE IMAGE — Call /predict API
// ============================================================
function analyzeImage() {
    if (!selectedFile) return;

    // Show loading
    loadingOverlay.classList.add('show');

    const formData = new FormData();
    formData.append('foto', selectedFile);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        loadingOverlay.classList.remove('show');

        if (data.status === 'success') {
            displayResults(data);
            // Gambar bounding box setelah gambar siap
            setTimeout(() => drawDetectionBoxes(data.hasil), 200);
        } else {
            alert(data.message || 'Terjadi kesalahan.');
        }
    })
    .catch(err => {
        loadingOverlay.classList.remove('show');
        console.error('Predict error:', err);
        alert('Gagal menghubungi server. Pastikan server Flask berjalan.');
    });
}

// ============================================================
// DRAW DETECTION BOXES on Canvas
// ============================================================
function drawDetectionBoxes(hasil) {
    if (!detectionCanvas || !previewImg) return;

    const img = previewImg;
    const canvas = detectionCanvas;
    
    // Tunggu gambar selesai dimuat
    if (!img.complete || img.naturalWidth === 0) {
        img.onload = () => drawDetectionBoxes(hasil);
        return;
    }

    // Karena model yang digunakan adalah Klasifikasi (bukan Deteksi Objek/YOLO),
    // kita tidak menggambar kotak deteksi buatan (hardcoded) di atas gambar.
    // Canvas hanya dibersihkan.
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Reset legend
    legendItems.innerHTML = '';

    // Buat legend untuk setiap komponen yang TERDETEKSI
    for (const [key, info] of Object.entries(hasil)) {
        const config = DETECTION_CONFIG[key];
        if (!config) continue;

        const isDetected = info.status === 'terdeteksi';

        // Tambahkan ke legend
        const legendEl = document.createElement('div');
        legendEl.className = `legend-item ${info.status}`;
        legendEl.innerHTML = `
            <div class="legend-swatch" style="background: ${config.color};"></div>
            ${config.icon} ${config.label}
            <span style="font-size: 0.75rem; opacity: 0.8;">
                ${isDetected ? '✅ ' + info.persen + '%' : '❌ ' + info.persen + '%'}
            </span>
        `;
        legendItems.appendChild(legendEl);
    }

    // Tampilkan legend
    detectionLegend.style.display = 'block';
}

// ============================================================
// CLEAR DETECTION OVERLAY
// ============================================================
function clearDetectionOverlay() {
    if (detectionCanvas) {
        const ctx = detectionCanvas.getContext('2d');
        ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
    }
    if (detectionLegend) {
        detectionLegend.style.display = 'none';
    }
}

// ============================================================
// DISPLAY RESULTS
// ============================================================
function displayResults(data) {
    const { hasil, total_terdeteksi, total_komponen } = data;

    // Score header
    resultScore.textContent = `${total_terdeteksi}/${total_komponen}`;
    resultScore.style.color = total_terdeteksi >= 4 ? 'var(--accent-green)' :
                              total_terdeteksi >= 2 ? 'var(--accent-yellow)' : 'var(--accent-red)';

    const messages = {
        5: 'Luar biasa! Bekalmu lengkap semua! 🎉',
        4: 'Hampir sempurna! Tinggal sedikit lagi! 💪',
        3: 'Lumayan bagus! Coba tambahkan yang kurang ya 😊',
        2: 'Masih perlu ditambah komponen gizinya 🤔',
        1: 'Ayo lengkapi bekalmu dengan lebih banyak gizi! 📝',
        0: 'Hmm, sepertinya bekalmu perlu dilengkapi lagi 😅'
    };
    resultSummary.textContent = messages[total_terdeteksi] || messages[0];

    // Result cards
    resultGrid.innerHTML = '';
    for (const [key, info] of Object.entries(hasil)) {
        const isOk = info.status === 'terdeteksi';
        const config = DETECTION_CONFIG[key] || {};
        const card = document.createElement('div');
        card.className = `result-item ${info.status}`;
        card.style.borderLeftWidth = '4px';
        card.style.borderLeftColor = config.color || '#ccc';
        card.innerHTML = `
            <div class="r-icon">${info.icon}</div>
            <div class="r-info" style="flex:1">
                <h4>${info.nama}</h4>
                <p style="font-size:0.75rem; color:var(--text-body); margin:2px 0 4px;">${info.deskripsi}</p>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill ${isOk ? 'high' : 'low'}" style="width: ${info.persen}%; background: ${config.color || ''}"></div>
                </div>
            </div>
            <div>
                <span class="r-status">${isOk ? '✅ ' + info.persen + '%' : '❌ ' + info.persen + '%'}</span>
            </div>
        `;
        resultGrid.appendChild(card);
    }

    // Show results
    resultContainer.classList.add('show');
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ============================================================
// RESET UPLOAD
// ============================================================
function resetUpload() {
    selectedFile = null;
    fotoInput.value = '';
    previewImg.src = '';
    uploadPlaceholder.style.display = 'block';
    previewArea.style.display = 'none';
    uploadZone.classList.remove('has-preview');
    analyzeBtn.style.display = 'none';
    resultContainer.classList.remove('show');
    clearDetectionOverlay();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ============================================================
// CAMERA CAPTURE
// ============================================================
let cameraStream = null;
let useFrontCamera = false;

/**
 * Open camera modal and start video stream.
 * Falls back to native file input with capture on mobile if getUserMedia fails.
 */
function openCamera() {
    const modal = document.getElementById('camera-modal');
    const video = document.getElementById('camera-video');

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        // Fallback for browsers without getUserMedia (e.g. older mobile browsers)
        // Use native camera input
        const cameraInput = document.getElementById('foto-input-camera');
        if (cameraInput) {
            cameraInput.click();
            cameraInput.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) handleFile(file);
            }, { once: true });
        }
        return;
    }

    startCameraStream(video).then(() => {
        modal.style.display = 'flex';
        document.body.style.overflow = 'hidden';
        showLiveControls();
    }).catch(err => {
        console.error('Camera error:', err);
        // Fallback to native camera
        const cameraInput = document.getElementById('foto-input-camera');
        if (cameraInput) {
            cameraInput.click();
            cameraInput.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) handleFile(file);
            }, { once: true });
        } else {
            alert('Kamera tidak bisa diakses. Pastikan Anda memberikan izin kamera.');
        }
    });
}

/**
 * Start camera stream with specified facing mode.
 */
async function startCameraStream(video) {
    // Stop existing stream if any
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
    }

    const constraints = {
        video: {
            facingMode: useFrontCamera ? 'user' : 'environment',
            width: { ideal: 1280 },
            height: { ideal: 960 }
        },
        audio: false
    };

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = cameraStream;
        await video.play();
    } catch (err) {
        // If specific facingMode fails, try without it
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 960 } },
            audio: false
        });
        video.srcObject = cameraStream;
        await video.play();
    }
}

/**
 * Close camera modal and stop all video tracks.
 */
function closeCamera() {
    const modal = document.getElementById('camera-modal');
    const video = document.getElementById('camera-video');
    const snapCanvas = document.getElementById('camera-snap-canvas');

    // Stop camera stream
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    video.srcObject = null;
    snapCanvas.style.display = 'none';

    modal.style.display = 'none';
    document.body.style.overflow = '';
    showLiveControls();
}

/**
 * Capture the current video frame onto the snapshot canvas.
 */
function capturePhoto() {
    const video = document.getElementById('camera-video');
    const snapCanvas = document.getElementById('camera-snap-canvas');
    const guides = document.getElementById('camera-guides');
    const hint = document.getElementById('camera-hint');

    // Set canvas to video dimensions
    snapCanvas.width = video.videoWidth;
    snapCanvas.height = video.videoHeight;

    // Draw current frame
    const ctx = snapCanvas.getContext('2d');
    ctx.drawImage(video, 0, 0, snapCanvas.width, snapCanvas.height);

    // Show canvas, hide video elements
    snapCanvas.style.display = 'block';
    guides.style.display = 'none';
    hint.style.display = 'none';

    // Switch to captured controls
    showCapturedControls();

    // Flash effect
    snapCanvas.style.animation = 'none';
    snapCanvas.offsetHeight; // trigger reflow
    snapCanvas.style.animation = 'fadeIn 0.3s ease';
}

/**
 * Go back to live camera view for retake.
 */
function retakePhoto() {
    const snapCanvas = document.getElementById('camera-snap-canvas');
    const guides = document.getElementById('camera-guides');
    const hint = document.getElementById('camera-hint');

    snapCanvas.style.display = 'none';
    guides.style.display = '';
    hint.style.display = '';

    showLiveControls();
}

/**
 * Use the captured photo — convert canvas to file and feed into upload flow.
 */
function usePhoto() {
    const snapCanvas = document.getElementById('camera-snap-canvas');

    snapCanvas.toBlob((blob) => {
        if (!blob) {
            alert('Gagal mengambil foto. Silakan coba lagi.');
            return;
        }

        // Create a File from the blob
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const file = new File([blob], `kamera_${timestamp}.jpg`, { type: 'image/jpeg' });

        // Feed into existing upload flow
        handleFile(file);

        // Close camera
        closeCamera();
    }, 'image/jpeg', 0.92);
}

/**
 * Switch between front and back camera.
 */
function switchCamera() {
    useFrontCamera = !useFrontCamera;
    const video = document.getElementById('camera-video');
    startCameraStream(video).catch(err => {
        console.error('Switch camera error:', err);
        useFrontCamera = !useFrontCamera; // revert
    });
}

/** Show live camera controls, hide captured controls. */
function showLiveControls() {
    const live = document.getElementById('camera-controls-live');
    const captured = document.getElementById('camera-controls-captured');
    if (live) live.style.display = 'flex';
    if (captured) captured.style.display = 'none';
}

/** Show captured controls, hide live controls. */
function showCapturedControls() {
    const live = document.getElementById('camera-controls-live');
    const captured = document.getElementById('camera-controls-captured');
    if (live) live.style.display = 'none';
    if (captured) captured.style.display = 'flex';
}

