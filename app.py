"""
NutriKids — Flask Backend Server
Sistem Klasifikasi Komponen Makanan Siswa Berbasis CNN (MobileNetV2)
Berdasarkan Bab 2.13, Class Diagram 'SistemWebController'

Database: SQLite (untuk development) — bisa diganti MySQL untuk production
"""

from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import numpy as np
import cv2
import sqlite3
from datetime import datetime, timedelta
import os
import random

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'nutrikids-secret-key-2026-ganti-di-production')

# ============================================================
# KONFIGURASI
# ============================================================
MODEL_PATH = 'model/nutrikids_final.h5'
MODEL_TFLITE_PATH = 'model/nutrikids_final.tflite'
THRESHOLD = 0.60
LABELS = ['makanan_pokok', 'lauk_pauk', 'sayur', 'buah', 'susu']
LABEL_DISPLAY = {
    'makanan_pokok': {'nama': 'Karbohidrat', 'icon': '🍚', 'deskripsi': 'Sumber tenaga untuk berlari, bermain, dan belajar.'},
    'lauk_pauk':     {'nama': 'Protein',     'icon': '🍗', 'deskripsi': 'Membantu tubuh tumbuh dan memperbaiki diri.'},
    'sayur':         {'nama': 'Sayuran',     'icon': '🥬', 'deskripsi': 'Menjaga tubuh tetap segar dan kuat.'},
    'buah':          {'nama': 'Buah-buahan', 'icon': '🍎', 'deskripsi': 'Vitamin alami untuk daya tahan tubuh.'},
    'susu':          {'nama': 'Susu',        'icon': '🥛', 'deskripsi': 'Kalsium untuk tulang dan gigi yang kuat.'},
}
UPLOAD_DIR = 'static/uploads'
DB_PATH = 'db/nutrikids.db'

# ============================================================
# DATABASE SETUP (SQLite)
# ============================================================
def get_db():
    """Buat koneksi ke SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # agar bisa akses kolom by name
    return conn


def init_db():
    """Inisialisasi database — buat tabel jika belum ada."""
    os.makedirs('db', exist_ok=True)
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS siswa (
            nis            TEXT PRIMARY KEY,
            nama_lengkap   TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS riwayat_konsumsi (
            id_riwayat          INTEGER PRIMARY KEY AUTOINCREMENT,
            nis_siswa           TEXT NOT NULL,
            tanggal_waktu       TEXT NOT NULL,
            path_foto           TEXT NOT NULL,
            status_karbohidrat  REAL NOT NULL,
            status_protein      REAL NOT NULL,
            status_sayur        REAL NOT NULL,
            status_buah         REAL NOT NULL,
            status_susu         REAL NOT NULL,
            FOREIGN KEY (nis_siswa) REFERENCES siswa(nis)
        )
    """)

    # Insert data siswa contoh (jika belum ada)
    cursor.execute("SELECT COUNT(*) FROM siswa")
    if cursor.fetchone()[0] == 0:
        cursor.executemany(
            "INSERT INTO siswa (nis, nama_lengkap) VALUES (?, ?)",
            [('1234', 'Yura'), ('1235', 'Made Arimbawa')]
        )
        print("[OK] Data siswa contoh ditambahkan: Yura (1234), Made Arimbawa (1235)")

    conn.commit()
    conn.close()
    print("[OK] Database SQLite siap di", DB_PATH)


# ============================================================
# MUAT MODEL (dengan fallback: TFLite -> TensorFlow -> Mock)
# ============================================================
model = None
tflite_interpreter = None
USE_MOCK = True
USE_TFLITE = False

# Coba TFLite dulu (ringan, cocok untuk hosting gratis)
try:
    import tflite_runtime.interpreter as tflite
    if os.path.exists(MODEL_TFLITE_PATH):
        tflite_interpreter = tflite.Interpreter(model_path=MODEL_TFLITE_PATH)
        tflite_interpreter.allocate_tensors()
        USE_TFLITE = True
        USE_MOCK = False
        print(f"[OK] Model TFLite dimuat dari {MODEL_TFLITE_PATH}")
    else:
        print(f"[WARN] Model TFLite tidak ditemukan di {MODEL_TFLITE_PATH}")
except ImportError:
    # Coba pakai tf.lite sebagai fallback
    try:
        import tensorflow as tf
        if os.path.exists(MODEL_TFLITE_PATH):
            tflite_interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE_PATH)
            tflite_interpreter.allocate_tensors()
            USE_TFLITE = True
            USE_MOCK = False
            print(f"[OK] Model TFLite dimuat via TensorFlow dari {MODEL_TFLITE_PATH}")
        elif os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            USE_MOCK = False
            print(f"[OK] Model H5 dimuat dari {MODEL_PATH}")
        else:
            print(f"[WARN] Model tidak ditemukan.")
            print("       Menggunakan MOCK PREDICTOR untuk development.")
    except ImportError:
        print("[WARN] TFLite Runtime dan TensorFlow tidak terinstall.")
        print("       Menggunakan MOCK PREDICTOR untuk development.")
    except Exception as e:
        print(f"[WARN] Error memuat model: {e}")
        print("       Menggunakan MOCK PREDICTOR untuk development.")
except Exception as e:
    print(f"[WARN] Error memuat model TFLite: {e}")
    print("       Menggunakan MOCK PREDICTOR untuk development.")


# ---------- Kelas PemrosesCitra (OpenCV) ----------
def preprocess_image(file_storage):
    """
    Pra-pemrosesan citra sesuai Bab 3.4.2:
    1. Baca file bytes -> decode gambar
    2. Konversi BGR -> RGB
    3. Resize ke 224x224 (standar MobileNetV2)
    4. Normalisasi warna /255
    """
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))          # resizeCitra()
    img = img.astype('float32') / 255.0        # normalisasiWarna()
    return np.expand_dims(img, axis=0)


def mock_predict():
    """Mock predictor untuk development — menghasilkan skor random yang realistis."""
    return np.array([
        random.uniform(0.3, 0.95),  # makanan_pokok
        random.uniform(0.2, 0.90),  # lauk_pauk
        random.uniform(0.1, 0.85),  # sayur
        random.uniform(0.05, 0.80), # buah
        random.uniform(0.05, 0.70), # susu
    ])


# ============================================================
# ROUTE: LOGIN (Gbr 3.10)
# ============================================================
@app.route('/')
def index():
    if 'nis' in session:
        return redirect(url_for('beranda'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        success_msg = request.args.get('success', '')
        return render_template('login.html', success=success_msg)

    nis = request.form.get('nis', '').strip()
    nama = request.form.get('nama', '').strip()

    if not nis or not nama:
        return render_template('login.html', error="Mohon isi NIS dan Nama Panggilan.")

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM siswa WHERE nis=? AND nama_lengkap=?", (nis, nama))
    siswa = cursor.fetchone()
    conn.close()

    if siswa:
        session['nis'] = nis
        session['nama'] = nama
        return redirect(url_for('beranda'))
    return render_template('login.html', error="Data Siswa Tidak Ditemukan")


# ============================================================
# ROUTE: DAFTAR AKUN (Registrasi Siswa Baru)
# ============================================================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('login.html', show_register=True)

    nis = request.form.get('nis', '').strip()
    nama = request.form.get('nama', '').strip()

    if not nis or not nama:
        return render_template('login.html', show_register=True,
                               error="Mohon isi NIS dan Nama Lengkap.")

    if not nis.isdigit():
        return render_template('login.html', show_register=True,
                               error="NIS harus berupa angka.")

    conn = get_db()
    cursor = conn.cursor()

    # Cek apakah NIS sudah terdaftar
    cursor.execute("SELECT * FROM siswa WHERE nis=?", (nis,))
    existing = cursor.fetchone()

    if existing:
        conn.close()
        return render_template('login.html', show_register=True,
                               error="NIS sudah terdaftar. Silakan langsung login.")

    # Daftarkan siswa baru
    try:
        cursor.execute("INSERT INTO siswa (nis, nama_lengkap) VALUES (?, ?)", (nis, nama))
        conn.commit()
        conn.close()
        print(f"[OK] Siswa baru terdaftar: {nama} ({nis})")
        return redirect(url_for('login', success=f'Pendaftaran berhasil! Silakan login dengan NIS {nis}.'))
    except Exception as e:
        conn.close()
        print(f"Register error: {e}")
        return render_template('login.html', show_register=True,
                               error="Terjadi kesalahan saat mendaftar. Silakan coba lagi.")


# ============================================================
# ROUTE: BERANDA (Gbr 3.11)
# ============================================================
@app.route('/beranda')
def beranda():
    if 'nis' not in session:
        return redirect(url_for('login'))

    # Ambil statistik mingguan
    stats = None
    try:
        conn = get_db()
        cursor = conn.cursor()
        week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute("""
            SELECT
                COUNT(*) as total_cek,
                AVG(status_karbohidrat) as avg_karbo,
                AVG(status_protein) as avg_protein,
                AVG(status_sayur) as avg_sayur,
                AVG(status_buah) as avg_buah,
                AVG(status_susu) as avg_susu
            FROM riwayat_konsumsi
            WHERE nis_siswa=? AND tanggal_waktu >= ?
        """, (session['nis'], week_ago))
        row = cursor.fetchone()
        if row and row['total_cek'] > 0:
            stats = dict(row)
        conn.close()
    except Exception as e:
        print(f"Stats error: {e}")

    return render_template('beranda.html',
                           nama=session['nama'],
                           stats=stats,
                           label_display=LABEL_DISPLAY,
                           threshold=THRESHOLD)


# ============================================================
# ROUTE: CEK BEKAL (Gbr 3.12-3.14)
# ============================================================
@app.route('/cek-bekal', methods=['GET'])
def cek_bekal_page():
    if 'nis' not in session:
        return redirect(url_for('login'))
    return render_template('cek_bekal.html', label_display=LABEL_DISPLAY)


@app.route('/predict', methods=['POST'])
def predict():
    if 'nis' not in session:
        return jsonify({'status': 'error', 'message': 'Silakan login dahulu'}), 401

    file = request.files.get('foto')
    if file is None:
        return jsonify({'status': 'error', 'message': 'Foto tidak ditemukan'}), 400

    # Prediksi
    if USE_MOCK:
        prediksi = mock_predict()
    elif USE_TFLITE:
        img_array = preprocess_image(file)
        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()
        tflite_interpreter.set_tensor(input_details[0]['index'], img_array)
        tflite_interpreter.invoke()
        prediksi = tflite_interpreter.get_tensor(output_details[0]['index'])[0]
    else:
        img_array = preprocess_image(file)
        prediksi = model.predict(img_array)[0]

    # Susun hasil per komponen
    hasil = {}
    for label, skor in zip(LABELS, prediksi):
        info = LABEL_DISPLAY[label]
        hasil[label] = {
            'skor': round(float(skor), 4),
            'persen': round(float(skor) * 100),
            'status': 'terdeteksi' if skor >= THRESHOLD else 'kurang',
            'nama': info['nama'],
            'icon': info['icon'],
            'deskripsi': info['deskripsi'],
        }

    # Simpan file foto
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filename = f"{session['nis']}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.seek(0)
    file.save(filepath)

    # Simpan ke SQLite
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO riwayat_konsumsi
            (nis_siswa, tanggal_waktu, path_foto, status_karbohidrat,
             status_protein, status_sayur, status_buah, status_susu)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session['nis'], datetime.now().strftime('%Y-%m-%d %H:%M:%S'), filepath,
            hasil['makanan_pokok']['skor'], hasil['lauk_pauk']['skor'],
            hasil['sayur']['skor'], hasil['buah']['skor'], hasil['susu']['skor']
        ))
        conn.commit()
        conn.close()
        print(f"[OK] Riwayat tersimpan untuk siswa {session['nis']}")
    except Exception as e:
        print(f"DB save error: {e}")

    # Hitung total komponen terdeteksi
    total_terdeteksi = sum(1 for h in hasil.values() if h['status'] == 'terdeteksi')

    return jsonify({
        'status': 'success',
        'hasil': hasil,
        'total_terdeteksi': total_terdeteksi,
        'total_komponen': len(LABELS),
        'foto_url': '/' + filepath.replace('\\', '/')
    })


# ============================================================
# ROUTE: HISTORI (Gbr 3.15)
# ============================================================
@app.route('/histori')
def histori():
    if 'nis' not in session:
        return redirect(url_for('login'))

    filter_waktu = request.args.get('filter', 'mingguan')  # harian/mingguan/bulanan
    days = {'harian': 1, 'mingguan': 7, 'bulanan': 30}.get(filter_waktu, 7)
    cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')

    data = []
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM riwayat_konsumsi
            WHERE nis_siswa=? AND tanggal_waktu >= ?
            ORDER BY tanggal_waktu DESC
        """, (session['nis'], cutoff))
        rows = cursor.fetchall()
        # Convert Row objects to dicts and parse datetime strings
        for row in rows:
            d = dict(row)
            d['tanggal_waktu'] = datetime.strptime(d['tanggal_waktu'], '%Y-%m-%d %H:%M:%S')
            data.append(d)
        conn.close()
    except Exception as e:
        print(f"Histori error: {e}")

    return render_template('histori.html',
                           riwayat=data,
                           filter_aktif=filter_waktu,
                           label_display=LABEL_DISPLAY,
                           threshold=THRESHOLD)


# ============================================================
# ROUTE: LOGOUT
# ============================================================
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ============================================================
# RUN SERVER
# ============================================================
# Inisialisasi saat import (untuk gunicorn di Render)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs('dataset/raw_images', exist_ok=True)
os.makedirs('dataset/images', exist_ok=True)
init_db()  # Auto-create database & tables

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    print(f"\n>>> NutriKids Server berjalan di http://127.0.0.1:{port}")
    print(f"    Mode: {'MOCK PREDICTOR' if USE_MOCK else 'MODEL CNN'}")
    app.run(host='0.0.0.0', port=port, debug=debug)
