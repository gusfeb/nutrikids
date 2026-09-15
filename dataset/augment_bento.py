"""
NutriKids - Augmentasi Data Sintetis (Bab 3.4.3)
=================================================
Pendekatan: CutMix / Copy-Paste Augmentation

Masalah:
  Dataset berisi foto makanan individual (nasi di piring, ayam goreng utuh, dll.)
  Tapi saat digunakan, user mengunggah foto BENTO BOX dari atas (top-down view)
  di mana semua komponen ada dalam satu frame. Ini menyebabkan Domain Shift.

Solusi:
  Membuat gambar sintetis yang mensimulasikan bento box / nampan makan:
  1. Ambil gambar acak dari masing-masing kategori (karbohidrat, lauk, sayur, buah, susu)
  2. Tempelkan (paste) ke dalam grid layout yang menyerupai bento box
  3. Tambahkan variasi (rotasi, brightness, noise) agar model lebih robust
  4. Label otomatis multi-label berdasarkan komponen yang dipakai

Referensi Akademis:
  - Dwibedi et al. (2017) "Cut, Paste and Learn" - ICCV
  - Ghiasi et al. (2021) "Simple Copy-Paste is a Strong Data Augmentation" - CVPR
"""

import os
import csv
import random
import numpy as np
import cv2
from pathlib import Path

# ============================================================
# KONFIGURASI
# ============================================================
DATASET_DIR = 'dataset/images'
LABELS_CSV = 'dataset/labels.csv'
OUTPUT_DIR = 'dataset/images'  # simpan di folder yang sama
LABEL_COLS = ['makanan_pokok', 'lauk_pauk', 'sayur', 'buah', 'susu']
IMG_SIZE = 224
NUM_SYNTHETIC = 500  # jumlah gambar sintetis yang dibuat

# Warna background yang realistis untuk bento/nampan
BG_COLORS = [
    (200, 200, 200),  # silver/stainless steel
    (180, 180, 175),  # abu-abu metalik
    (220, 215, 200),  # krem/putih pucat
    (190, 185, 170),  # abu krem
    (210, 200, 190),  # putih kecoklatan
]

# ============================================================
# TAHAP 1: INDEKS GAMBAR PER KATEGORI
# ============================================================
def load_category_index():
    """Baca labels.csv dan kelompokkan filename berdasarkan kategori."""
    import pandas as pd
    df = pd.read_csv(LABELS_CSV)
    
    category_files = {}
    for col in LABEL_COLS:
        files = df[df[col] == 1]['filename'].tolist()
        # Filter hanya file yang benar-benar ada
        files = [f for f in files if os.path.exists(os.path.join(DATASET_DIR, f))]
        category_files[col] = files
        print(f"  [{col}] {len(files)} gambar ditemukan")
    
    return category_files


# ============================================================
# TAHAP 2: FUNGSI AUGMENTASI INDIVIDUAL
# ============================================================
def load_and_augment(filepath, target_size):
    """
    Muat gambar, resize ke target_size, dan terapkan augmentasi acak.
    Augmentasi: rotasi, flip, brightness, contrast, blur ringan.
    """
    img = cv2.imread(filepath)
    if img is None:
        return None
    
    img = cv2.resize(img, (target_size, target_size))
    
    # Random horizontal flip (50% chance)
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    
    # Random rotation (-15 to +15 derajat)
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((target_size // 2, target_size // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (target_size, target_size),
                         borderMode=cv2.BORDER_REFLECT_101)
    
    # Random brightness & contrast
    alpha = random.uniform(0.8, 1.2)  # contrast
    beta = random.randint(-20, 20)     # brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    # Random blur ringan (simulasi camera focus)
    if random.random() > 0.7:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    
    return img


# ============================================================
# TAHAP 3: BENTO BOX GRID GENERATOR
# ============================================================
def create_bento_grid(images, labels_active):
    """
    Susun gambar-gambar komponen makanan ke dalam layout grid
    yang menyerupai bento box / nampan makan sekolah.
    
    Layout bervariasi antara 2x2, 2x3, atau format L-shape
    agar model belajar berbagai tata letak.
    """
    canvas_size = IMG_SIZE
    n = len(images)
    
    # Pilih warna background acak (simulasi nampan/bento)
    bg_color = random.choice(BG_COLORS)
    # Tambahkan noise ke warna agar lebih natural
    bg_color = tuple(max(0, min(255, c + random.randint(-10, 10))) for c in bg_color)
    canvas = np.full((canvas_size, canvas_size, 3), bg_color, dtype=np.uint8)
    
    # Tambahkan tekstur halus ke background (simulasi permukaan metal)
    noise = np.random.randint(-8, 8, canvas.shape, dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    if n == 2:
        # Layout 1x2 atau 2x1
        if random.random() > 0.5:
            # Side by side
            cell_w = canvas_size // 2 - 4
            cell_h = canvas_size - 8
            positions = [(4, 4, cell_w, cell_h), (cell_w + 8, 4, cell_w, cell_h)]
        else:
            # Top-bottom
            cell_w = canvas_size - 8
            cell_h = canvas_size // 2 - 4
            positions = [(4, 4, cell_w, cell_h), (4, cell_h + 8, cell_w, cell_h)]
    elif n == 3:
        # Layout: 1 besar di kiri, 2 kecil di kanan
        big_w = canvas_size // 2 - 4
        big_h = canvas_size - 8
        small_w = canvas_size // 2 - 8
        small_h = canvas_size // 2 - 6
        positions = [
            (4, 4, big_w, big_h),
            (big_w + 8, 4, small_w, small_h),
            (big_w + 8, small_h + 10, small_w, small_h)
        ]
    elif n == 4:
        # Layout grid 2x2
        cell_w = canvas_size // 2 - 6
        cell_h = canvas_size // 2 - 6
        positions = [
            (4, 4, cell_w, cell_h),
            (cell_w + 8, 4, cell_w, cell_h),
            (4, cell_h + 8, cell_w, cell_h),
            (cell_w + 8, cell_h + 8, cell_w, cell_h)
        ]
    else:  # n == 5
        # Layout: 3 atas + 2 bawah (seperti bento box nyata)
        top_w = canvas_size // 3 - 5
        top_h = canvas_size // 2 - 6
        bot_w = canvas_size // 2 - 6
        bot_h = canvas_size // 2 - 6
        positions = [
            (4, 4, top_w, top_h),
            (top_w + 8, 4, top_w, top_h),
            (2 * (top_w + 4), 4, top_w, top_h),
            (4, top_h + 8, bot_w, bot_h),
            (bot_w + 8, top_h + 8, bot_w, bot_h)
        ]
    
    # Paste setiap komponen ke posisinya
    for i, img in enumerate(images):
        if i >= len(positions):
            break
        x, y, w, h = positions[i]
        resized = cv2.resize(img, (w, h))
        
        # Tambahkan border tipis (simulasi sekat bento)
        cv2.rectangle(canvas, (x-1, y-1), (x+w, y+h), 
                     (160, 160, 155), 1)
        
        canvas[y:y+h, x:x+w] = resized
    
    # Post-processing: sedikit blur keseluruhan (simulasi kamera HP)
    if random.random() > 0.6:
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    
    # Random overall brightness shift
    shift = random.randint(-15, 15)
    canvas = np.clip(canvas.astype(np.int16) + shift, 0, 255).astype(np.uint8)
    
    return canvas


# ============================================================
# TAHAP 4: GENERATOR UTAMA
# ============================================================
def generate_synthetic_bento(category_files, num_images):
    """
    Generate gambar sintetis bento box dengan komposisi acak.
    Setiap gambar berisi 2-5 komponen makanan yang dipilih secara random.
    """
    new_rows = []
    generated = 0
    
    for i in range(num_images):
        # Tentukan berapa komponen dalam bento ini (2-5)
        num_components = random.randint(2, 5)
        
        # Pilih kategori mana saja yang muncul
        chosen_labels = random.sample(LABEL_COLS, num_components)
        
        # Ambil satu gambar acak dari setiap kategori terpilih
        component_images = []
        valid = True
        for label in chosen_labels:
            if not category_files[label]:
                valid = False
                break
            fname = random.choice(category_files[label])
            fpath = os.path.join(DATASET_DIR, fname)
            
            # Ukuran cell tergantung jumlah komponen
            cell_size = max(60, IMG_SIZE // num_components + 20)
            img = load_and_augment(fpath, cell_size)
            if img is None:
                valid = False
                break
            component_images.append(img)
        
        if not valid:
            continue
        
        # Buat komposisi bento
        bento_img = create_bento_grid(component_images, chosen_labels)
        
        # Simpan
        out_name = f"synth_bento_{i:04d}.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, bento_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        # Buat label multi-label
        row = {'filename': out_name}
        for col in LABEL_COLS:
            row[col] = 1 if col in chosen_labels else 0
        new_rows.append(row)
        generated += 1
        
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{num_images}] gambar sintetis dibuat...")
    
    return new_rows, generated


# ============================================================
# TAHAP 5: SINGLE-ITEM AUGMENTASI (untuk Buah & Susu yang minim)
# ============================================================
def augment_minority_classes(category_files, target_per_class=600):
    """
    Untuk kategori minoritas (buah=288, susu=724), buat variasi augmentasi
    tambahan agar jumlahnya mendekati kategori mayoritas.
    Teknik: Random Crop, Color Jitter, Rotation, Perspective Transform.
    """
    new_rows = []
    
    for col in LABEL_COLS:
        current_count = len(category_files[col])
        if current_count >= target_per_class:
            print(f"  [{col}] sudah cukup ({current_count} >= {target_per_class}), skip.")
            continue
        
        need = target_per_class - current_count
        print(f"  [{col}] perlu {need} gambar tambahan (dari {current_count})...")
        
        for j in range(need):
            src_fname = random.choice(category_files[col])
            src_path = os.path.join(DATASET_DIR, src_fname)
            
            img = cv2.imread(src_path)
            if img is None:
                continue
            
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            # === Augmentasi Agresif ===
            
            # 1. Random perspective transform (simulasi sudut kamera berbeda)
            pts1 = np.float32([[0, 0], [IMG_SIZE, 0], [0, IMG_SIZE], [IMG_SIZE, IMG_SIZE]])
            offset = 15
            pts2 = np.float32([
                [random.randint(0, offset), random.randint(0, offset)],
                [IMG_SIZE - random.randint(0, offset), random.randint(0, offset)],
                [random.randint(0, offset), IMG_SIZE - random.randint(0, offset)],
                [IMG_SIZE - random.randint(0, offset), IMG_SIZE - random.randint(0, offset)]
            ])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            img = cv2.warpPerspective(img, M, (IMG_SIZE, IMG_SIZE),
                                      borderMode=cv2.BORDER_REFLECT_101)
            
            # 2. Color jitter (HSV space)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-10, 10)) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] + random.randint(-30, 30), 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] + random.randint(-30, 30), 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            
            # 3. Random rotation
            angle = random.uniform(-25, 25)
            M_rot = cv2.getRotationMatrix2D((IMG_SIZE // 2, IMG_SIZE // 2), angle, 1.0)
            img = cv2.warpAffine(img, M_rot, (IMG_SIZE, IMG_SIZE),
                                 borderMode=cv2.BORDER_REFLECT_101)
            
            # 4. Random horizontal/vertical flip
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
            if random.random() > 0.8:
                img = cv2.flip(img, 0)
            
            # 5. Random Gaussian noise
            if random.random() > 0.6:
                noise = np.random.normal(0, random.randint(5, 15), img.shape).astype(np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Simpan
            out_name = f"aug_{col}_{j:04d}.jpg"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            
            row = {'filename': out_name}
            for label_col in LABEL_COLS:
                row[label_col] = 1 if label_col == col else 0
            new_rows.append(row)
    
    return new_rows


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("NutriKids - Augmentasi Data Sintetis (CutMix / Copy-Paste)")
    print("=" * 60)
    
    print("\n[1/4] Memuat indeks gambar per kategori...")
    category_files = load_category_index()
    
    print(f"\n[2/4] Membuat {NUM_SYNTHETIC} gambar sintetis bento box...")
    bento_rows, bento_count = generate_synthetic_bento(category_files, NUM_SYNTHETIC)
    print(f"  Berhasil: {bento_count} gambar bento sintetis")
    
    print(f"\n[3/4] Augmentasi kelas minoritas (buah, susu)...")
    minority_rows = augment_minority_classes(category_files, target_per_class=600)
    print(f"  Berhasil: {len(minority_rows)} gambar augmentasi minoritas")
    
    print(f"\n[4/4] Menambahkan {len(bento_rows) + len(minority_rows)} entri ke labels.csv...")
    all_new_rows = bento_rows + minority_rows
    with open(LABELS_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename'] + LABEL_COLS)
        writer.writerows(all_new_rows)
    
    # Cetak distribusi final
    import pandas as pd
    df_final = pd.read_csv(LABELS_CSV)
    print(f"\n{'=' * 60}")
    print(f"HASIL AKHIR")
    print(f"{'=' * 60}")
    print(f"Total gambar: {len(df_final)}")
    print(f"Distribusi label:")
    print(df_final[LABEL_COLS].astype(int).sum())
    
    # Hitung gambar sintetis bento
    synth_count = len(df_final[df_final['filename'].str.startswith('synth_bento')])
    aug_count = len(df_final[df_final['filename'].str.startswith('aug_')])
    print(f"\n  Gambar bento sintetis: {synth_count}")
    print(f"  Gambar augmentasi minoritas: {aug_count}")
    print(f"\nSelesai! Jalankan train_model.py untuk melatih ulang model.")
