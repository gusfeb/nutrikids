"""
NutriKids — Dataset Labeling Tool
Aplikasi web sederhana (Flask) untuk mempercepat proses pelabelan foto dataset.

Jalankan:
python dataset/labeling_tool.py
Lalu buka: http://127.0.0.1:5001
"""

from flask import Flask, render_template_string, request, redirect, url_for, send_from_directory
import os
import csv

app = Flask(__name__)

# Konfigurasi Path
BASE_DIR = os.path.dirname(__file__)
IMG_DIR = os.path.join(BASE_DIR, 'images')
LABELS_FILE = os.path.join(BASE_DIR, 'labels.csv')
LABEL_COLS = ['makanan_pokok', 'lauk_pauk', 'sayur', 'buah', 'susu']

# Inisialisasi file CSV jika belum ada
if not os.path.exists(LABELS_FILE):
    with open(LABELS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename'] + LABEL_COLS)

def read_labels():
    labels = {}
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[row['filename']] = {col: int(row.get(col, 0)) for col in LABEL_COLS}
    return labels

def save_labels(labels_dict):
    with open(LABELS_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename'] + LABEL_COLS)
        writer.writeheader()
        for filename, data in labels_dict.items():
            row = {'filename': filename}
            row.update(data)
            writer.writerow(row)

# ============================================================
# TEMPLATE HTML (Single File)
# ============================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>NutriKids Labeling Tool</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #FFF8ED; color: #1B3A4B; margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center; }
        .container { background: white; padding: 30px; border-radius: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); max-width: 800px; width: 100%; text-align: center; }
        h1 { margin-top: 0; color: #FF6B35; }
        .progress { font-size: 1.2rem; font-weight: bold; margin-bottom: 20px; color: #8DA4B5; }
        .image-container { display: flex; justify-content: center; margin-bottom: 30px; }
        .image-container img { max-width: 100%; max-height: 400px; border-radius: 12px; border: 2px solid #E8EDF2; }
        .checkbox-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 30px; text-align: left; }
        .checkbox-item { background: #E3F6FD; padding: 15px; border-radius: 10px; display: flex; align-items: center; gap: 10px; cursor: pointer; transition: 0.2s; border: 2px solid transparent; }
        .checkbox-item:hover { background: #B3E5FC; }
        .checkbox-item.checked { background: #E8F5E9; border-color: #66BB6A; }
        input[type="checkbox"] { transform: scale(1.5); cursor: pointer; }
        .label-name { font-weight: bold; font-size: 1.1rem; }
        .buttons { display: flex; justify-content: space-between; gap: 15px; }
        .btn { padding: 12px 25px; border: none; border-radius: 8px; font-weight: bold; font-size: 1rem; cursor: pointer; transition: 0.2s; text-decoration: none; display: inline-block; }
        .btn-prev { background: #E8EDF2; color: #1B3A4B; }
        .btn-prev:hover { background: #CFD8DC; }
        .btn-next { background: #FF6B35; color: white; flex-grow: 1; }
        .btn-next:hover { background: #E55A28; }
        .alert { background: #E8F5E9; color: #2E7D32; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🍱 NutriKids Labeling Tool</h1>
        
        {% if total == 0 %}
            <p>Tidak ada gambar di folder <b>dataset/images/</b>.</p>
            <p>Silakan masukkan foto bekal ke folder tersebut terlebih dahulu.</p>
        {% else %}
            <div class="progress">Gambar {{ index + 1 }} dari {{ total }}</div>
            <div style="margin-bottom: 15px; font-family: monospace;">{{ filename }}</div>
            
            <div class="image-container">
                <img src="{{ url_for('serve_image', filename=filename) }}" alt="Food Image">
            </div>

            <form method="POST" action="{{ url_for('save', index=index) }}">
                <div class="checkbox-grid">
                    {% for col in label_cols %}
                    <label class="checkbox-item {% if current_labels[col] == 1 %}checked{% endif %}" id="label-{{ col }}">
                        <input type="checkbox" name="{{ col }}" value="1" {% if current_labels[col] == 1 %}checked{% endif %}
                               onchange="document.getElementById('label-{{ col }}').classList.toggle('checked')">
                        <span class="label-name">{{ col.replace('_', ' ').title() }}</span>
                    </label>
                    {% endfor %}
                </div>
                
                <div class="buttons">
                    {% if index > 0 %}
                        <a href="{{ url_for('index', idx=index-1) }}" class="btn btn-prev">⬅️ Sebelumnya</a>
                    {% else %}
                        <div style="width: 140px;"></div>
                    {% endif %}
                    
                    <button type="submit" class="btn btn-next">
                        {% if index < total - 1 %} Simpan & Lanjut ➡️ {% else %} Simpan Selesai ✅ {% endif %}
                    </button>
                </div>
            </form>
        {% endif %}
    </div>
</body>
</html>
"""

# ============================================================
# ROUTES
# ============================================================
@app.route('/image/<filename>')
def serve_image(filename):
    return send_from_directory(IMG_DIR, filename)


@app.route('/')
def index():
    # Ambil semua gambar
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)
        
    img_files = sorted([f for f in os.listdir(IMG_DIR) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    
    idx = request.args.get('idx', default=0, type=int)
    
    if not img_files:
        return render_template_string(HTML_TEMPLATE, total=0)
        
    # Pastikan index valid
    if idx < 0: idx = 0
    if idx >= len(img_files): idx = len(img_files) - 1
        
    filename = img_files[idx]
    
    # Ambil label saat ini
    labels = read_labels()
    current_labels = labels.get(filename, {col: 0 for col in LABEL_COLS})
    
    return render_template_string(
        HTML_TEMPLATE, 
        filename=filename, 
        index=idx, 
        total=len(img_files),
        label_cols=LABEL_COLS,
        current_labels=current_labels
    )


@app.route('/save/<int:index>', methods=['POST'])
def save(index):
    img_files = sorted([f for f in os.listdir(IMG_DIR) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
                       
    if index < 0 or index >= len(img_files):
        return redirect(url_for('index'))
        
    filename = img_files[index]
    
    # Ambil input dari form
    new_data = {}
    for col in LABEL_COLS:
        new_data[col] = 1 if request.form.get(col) == '1' else 0
        
    # Update CSV
    labels = read_labels()
    labels[filename] = new_data
    save_labels(labels)
    
    # Lanjut ke gambar berikutnya
    if index < len(img_files) - 1:
        return redirect(url_for('index', idx=index + 1))
    else:
        # Kembali ke gambar terakhir jika sudah selesai
        return redirect(url_for('index', idx=index))


if __name__ == '__main__':
    print("==================================================")
    print("NutriKids Labeling Tool Aktif")
    print("==================================================")
    print("Buka browser dan akses: http://127.0.0.1:5001")
    print("==================================================")
    # Gunakan port 5001 agar tidak bentrok dengan aplikasi utama (5000)
    app.run(debug=True, port=5001)
