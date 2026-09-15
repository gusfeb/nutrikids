---
title: NutriKids
emoji: 🍱
colorFrom: yellow
colorTo: orange
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# 🍱 NutriKids

**Sistem Klasifikasi Komponen Makanan Siswa Berbasis CNN (MobileNetV2)**

Proyek seminar Ni Putu Ayu Sri Ratnasari — INSTIKI Denpasar

## Fitur
- 📸 Upload foto bekal makanan
- 🤖 Klasifikasi otomatis 5 komponen gizi (Karbohidrat, Protein, Sayur, Buah, Susu)
- 📊 Histori konsumsi dengan grafik (harian/mingguan/bulanan)
- 🎨 UI ramah anak dengan desain modern

## Cara Paling Mudah Menjalankan Sistem (Rekomendasi)

Untuk menjalankan sistem dengan lebih praktis (khusus Windows), Anda cukup melakukan **Klik Ganda (Double-Click)** pada file:
👉 **`run.bat`**

File ini akan otomatis:
1. Mengaktifkan *virtual environment* (`venv` atau `venv311`).
2. Menginstal *dependencies* yang kurang secara otomatis.
3. Menjalankan server aplikasi web.

Setelah server berjalan, aplikasi akan terbuka di: 👉 **http://127.0.0.1:5000**

---

## Cara Menjalankan Sistem (Manual via Terminal)

Sistem ini menggunakan **SQLite** secara bawaan untuk memudahkan tahap *development*. Anda tidak perlu melakukan setup database secara manual karena sistem akan otomatis membuat tabel dan mengisi data contoh saat dijalankan pertama kali.

### Langkah 1: Buka Terminal
Buka terminal (Command Prompt, PowerShell, atau terminal di VS Code) dan pastikan Anda berada di dalam folder proyek `nutrikids`.

### Langkah 2: Aktifkan Virtual Environment
Aktifkan *virtual environment* Python agar *library* terisolasi:
```bash
venv\Scripts\activate
# Catatan: Jika menggunakan Python 3.11, Anda bisa menjalankan: venv311\Scripts\activate
```

### Langkah 3: Install Dependencies (Jika Belum)
Jika ini pertama kalinya proyek dijalankan, instal semua modul yang dibutuhkan:
```bash
pip install -r requirements.txt
```

### Langkah 4: Jalankan Server Aplikasi
Jalankan file backend utama:
```bash
python app.py
```

### Langkah 5: Buka Aplikasi di Browser
Setelah melihat pesan bahwa server telah berjalan di terminal, buka browser Anda (Chrome/Edge/Firefox) lalu kunjungi alamat:
👉 **http://127.0.0.1:5000**

### 🔑 Data Login Uji Coba
Gunakan salah satu data berikut untuk masuk ke dalam sistem:
- Nama Panggilan: `Yura` | NIS: `1234`
- Nama Panggilan: `Made Arimbawa` | NIS: `1235`

### 📈 Menjalankan Training Monitor UI (Khusus Pelatihan AI)
Jika Anda ingin melatih ulang model AI dan melihat grafik serta proses pelatihan secara *real-time*, Anda bisa menggunakan *Training Monitor UI*.
Buka **terminal baru** di dalam folder proyek, pastikan *virtual environment* aktif, lalu jalankan:
```bash
python model\training_ui.py
```
Akses *dashboard* pelatihannya di browser pada alamat: 👉 **http://127.0.0.1:5002**

## Struktur Folder
```
nutrikids/
├── app.py                  # Flask server
├── model/
│   ├── train_model.py      # Training pipeline MobileNetV2
│   └── evaluate_model.py   # Evaluasi model
├── dataset/
│   └── labels.csv          # Template label multi-label
├── db/
│   └── schema.sql          # Schema MySQL
├── static/
│   ├── css/style.css
│   └── js/main.js
├── templates/
│   ├── login.html
│   ├── beranda.html
│   ├── cek_bekal.html
│   └── histori.html
└── requirements.txt
```

## Tech Stack
- **Backend:** Flask + Python 3.10+
- **ML:** TensorFlow/Keras, MobileNetV2
- **Database:** SQLite (Development) / MySQL (Production)
- **Frontend:** HTML, CSS, JavaScript, Chart.js
