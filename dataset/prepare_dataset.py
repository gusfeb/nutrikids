"""
NutriKids — Dataset Preparation Helper
Membantu menyiapkan dataset untuk training model MobileNetV2.

Fitur:
1. Download contoh gambar makanan dari internet
2. Resize gambar ke 224x224
3. Generate template labels.csv
4. Validasi dataset sebelum training

Jalankan: python dataset/prepare_dataset.py
"""

import os
import csv
import shutil
from pathlib import Path

# Konfigurasi
RAW_DIR = os.path.join(os.path.dirname(__file__), 'raw_images')
IMG_DIR = os.path.join(os.path.dirname(__file__), 'images')
LABELS_FILE = os.path.join(os.path.dirname(__file__), 'labels.csv')
IMG_SIZE = (224, 224)
LABEL_COLS = ['makanan_pokok', 'lauk_pauk', 'sayur', 'buah', 'susu']


def resize_all_images():
    """
    Resize semua gambar di raw_images/ ke 224x224 dan simpan di images/.
    Sesuai Bab 3.4.2 — standar input MobileNetV2.
    """
    try:
        import cv2
    except ImportError:
        print("[ERROR] OpenCV belum terinstall. Jalankan: pip install opencv-python")
        return

    os.makedirs(IMG_DIR, exist_ok=True)
    raw_files = [f for f in os.listdir(RAW_DIR)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]

    if not raw_files:
        print(f"[WARN] Tidak ada gambar di {RAW_DIR}")
        print("       Letakkan foto makanan di folder raw_images/ lalu jalankan ulang.")
        return

    print(f"Memproses {len(raw_files)} gambar dari raw_images/ ...")
    success = 0
    for fname in raw_files:
        try:
            src = os.path.join(RAW_DIR, fname)
            img = cv2.imread(src)
            if img is None:
                print(f"  [SKIP] {fname} — tidak bisa dibaca")
                continue

            img_resized = cv2.resize(img, IMG_SIZE)

            # Pastikan format output .jpg
            out_name = Path(fname).stem + '.jpg'
            dst = os.path.join(IMG_DIR, out_name)
            cv2.imwrite(dst, img_resized)
            success += 1
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}")

    print(f"[OK] {success}/{len(raw_files)} gambar berhasil di-resize ke {IMG_SIZE}")
    print(f"     Tersimpan di: {IMG_DIR}")


def generate_label_template():
    """
    Generate template labels.csv dari gambar yang ada di images/.
    Semua label diisi 0 — Anda perlu mengeditnya manual.
    """
    img_files = sorted([f for f in os.listdir(IMG_DIR)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    if not img_files:
        print(f"[WARN] Tidak ada gambar di {IMG_DIR}")
        print("       Jalankan resize_all_images() dulu atau letakkan gambar langsung.")
        return

    # Cek apakah labels.csv sudah ada
    existing_labels = {}
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_labels[row['filename']] = row

    # Buat/update CSV
    with open(LABELS_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename'] + LABEL_COLS)
        writer.writeheader()

        new_count = 0
        for fname in img_files:
            if fname in existing_labels:
                writer.writerow(existing_labels[fname])
            else:
                row = {'filename': fname}
                for col in LABEL_COLS:
                    row[col] = 0
                writer.writerow(row)
                new_count += 1

    total = len(img_files)
    print(f"[OK] labels.csv diperbarui: {total} gambar total ({new_count} baru)")
    print(f"     File: {LABELS_FILE}")
    if new_count > 0:
        print(f"\n     >>> PENTING: Edit labels.csv dan isi label 0/1 untuk setiap gambar! <<<")
        print(f"     Buka di Excel/Google Sheets, isi kolom makanan_pokok s/d susu.")
        print(f"     1 = komponen terlihat di foto, 0 = tidak ada")


def validate_dataset():
    """Validasi bahwa dataset siap untuk training."""
    print("\n" + "=" * 50)
    print("VALIDASI DATASET")
    print("=" * 50)

    errors = []
    warnings = []

    # 1. Cek folder images
    if not os.path.exists(IMG_DIR):
        errors.append(f"Folder {IMG_DIR} tidak ditemukan")
    else:
        img_files = [f for f in os.listdir(IMG_DIR)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(img_files) == 0:
            errors.append("Tidak ada gambar di folder images/")
        elif len(img_files) < 50:
            warnings.append(f"Hanya {len(img_files)} gambar — disarankan minimal 200+ untuk hasil baik")
        else:
            print(f"  [OK] {len(img_files)} gambar ditemukan")

    # 2. Cek labels.csv
    if not os.path.exists(LABELS_FILE):
        errors.append(f"File {LABELS_FILE} tidak ditemukan")
    else:
        with open(LABELS_FILE, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if len(rows) == 0:
            errors.append("labels.csv kosong")
        else:
            print(f"  [OK] {len(rows)} baris di labels.csv")

            # Cek apakah semua gambar ada labelnya
            labeled_files = set(r['filename'] for r in rows)
            if os.path.exists(IMG_DIR):
                img_files_set = set(f for f in os.listdir(IMG_DIR)
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png')))
                unlabeled = img_files_set - labeled_files
                if unlabeled:
                    warnings.append(f"{len(unlabeled)} gambar belum punya label di CSV")

                missing_img = labeled_files - img_files_set
                if missing_img:
                    errors.append(f"{len(missing_img)} file di CSV tidak ada gambarnya: {list(missing_img)[:3]}")

            # Cek apakah ada label yang masih semua 0
            all_zero = sum(1 for r in rows
                         if all(r.get(c, '0') == '0' for c in LABEL_COLS))
            if all_zero > 0:
                warnings.append(f"{all_zero} gambar masih semua labelnya 0 — belum dilabeli?")

            # Distribusi label
            print("\n  Distribusi label:")
            for col in LABEL_COLS:
                count = sum(1 for r in rows if r.get(col, '0') == '1')
                pct = count / len(rows) * 100 if rows else 0
                bar = '#' * int(pct / 5)
                print(f"    {col:20s}: {count:4d} ({pct:5.1f}%) {bar}")

    # Ringkasan
    print()
    if errors:
        for e in errors:
            print(f"  [ERROR] {e}")
    if warnings:
        for w in warnings:
            print(f"  [WARN]  {w}")

    if not errors:
        print("\n  >>> Dataset SIAP untuk training! <<<")
        return True
    else:
        print("\n  >>> Perbaiki error di atas sebelum training <<<")
        return False


def create_sample_images():
    """
    Buat beberapa gambar contoh (placeholder) untuk testing pipeline.
    Gambar ini bukan foto makanan asli — hanya untuk memastikan pipeline training berjalan.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("[ERROR] OpenCV/NumPy belum terinstall.")
        return

    os.makedirs(IMG_DIR, exist_ok=True)

    # Warna per komponen makanan
    colors = {
        'nasi': (255, 255, 240),       # putih kekuningan
        'ayam': (180, 130, 80),         # cokelat
        'sayur': (80, 180, 80),         # hijau
        'buah': (220, 80, 80),          # merah
        'susu': (240, 240, 255),        # putih kebiruan
    }

    sample_configs = [
        ('sample_001.jpg', {'nasi': True, 'ayam': True, 'sayur': True, 'buah': False, 'susu': False},
         [1, 1, 1, 0, 0]),
        ('sample_002.jpg', {'nasi': True, 'ayam': True, 'sayur': True, 'buah': True, 'susu': True},
         [1, 1, 1, 1, 1]),
        ('sample_003.jpg', {'nasi': False, 'ayam': True, 'sayur': False, 'buah': True, 'susu': False},
         [0, 1, 0, 1, 0]),
        ('sample_004.jpg', {'nasi': True, 'ayam': False, 'sayur': True, 'buah': False, 'susu': True},
         [1, 0, 1, 0, 1]),
        ('sample_005.jpg', {'nasi': True, 'ayam': True, 'sayur': False, 'buah': False, 'susu': False},
         [1, 1, 0, 0, 0]),
        ('sample_006.jpg', {'nasi': True, 'ayam': True, 'sayur': True, 'buah': True, 'susu': False},
         [1, 1, 1, 1, 0]),
        ('sample_007.jpg', {'nasi': False, 'ayam': False, 'sayur': True, 'buah': True, 'susu': True},
         [0, 0, 1, 1, 1]),
        ('sample_008.jpg', {'nasi': True, 'ayam': False, 'sayur': False, 'buah': True, 'susu': False},
         [1, 0, 0, 1, 0]),
        ('sample_009.jpg', {'nasi': True, 'ayam': True, 'sayur': True, 'buah': False, 'susu': True},
         [1, 1, 1, 0, 1]),
        ('sample_010.jpg', {'nasi': False, 'ayam': True, 'sayur': True, 'buah': False, 'susu': False},
         [0, 1, 1, 0, 0]),
    ]

    labels_rows = []
    for fname, components, labels in sample_configs:
        # Buat gambar 224x224 dengan blok warna
        img = np.ones((224, 224, 3), dtype=np.uint8) * 200  # background abu-abu terang

        positions = [(10, 10, 100, 100), (114, 10, 100, 100),
                     (10, 114, 100, 100), (114, 114, 100, 100),
                     (62, 62, 100, 100)]

        for (comp_name, present), (x, y, w, h) in zip(components.items(), positions):
            if present:
                color = colors[comp_name]
                # Tambah sedikit noise agar tidak identik
                noise = np.random.randint(-20, 20, (h, w, 3))
                block = np.clip(np.full((h, w, 3), color, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
                img[y:y+h, x:x+w] = block

        filepath = os.path.join(IMG_DIR, fname)
        cv2.imwrite(filepath, img)
        labels_rows.append({'filename': fname, **dict(zip(LABEL_COLS, labels))})

    # Simpan labels
    with open(LABELS_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename'] + LABEL_COLS)
        writer.writeheader()
        writer.writerows(labels_rows)

    print(f"[OK] {len(sample_configs)} gambar contoh dibuat di {IMG_DIR}")
    print(f"[OK] labels.csv diperbarui dengan label yang benar")
    print(f"\n     Gambar ini hanya placeholder untuk testing pipeline.")
    print(f"     Ganti dengan foto makanan asli untuk hasil akurat!")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 50)
    print("NutriKids — Dataset Preparation Helper")
    print("=" * 50)

    print("\nPilih aksi:")
    print("  1. Buat gambar contoh (untuk test pipeline)")
    print("  2. Resize gambar dari raw_images/")
    print("  3. Generate template labels.csv")
    print("  4. Validasi dataset")
    print("  5. Semua (1 + 4)")

    choice = input("\nPilihan [1-5]: ").strip()

    if choice == '1':
        create_sample_images()
    elif choice == '2':
        resize_all_images()
    elif choice == '3':
        generate_label_template()
    elif choice == '4':
        validate_dataset()
    elif choice == '5':
        create_sample_images()
        validate_dataset()
    else:
        print("Pilihan tidak valid.")
