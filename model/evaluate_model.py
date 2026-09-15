"""
NutriKids — Evaluasi Model
Menghasilkan Confusion Matrix & Classification Report per label
Berdasarkan Bab 3.8.1 Proposal Seminar
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# KONFIGURASI
# ============================================================
THRESHOLD = 0.60   # sesuai Bab 2.18
LABEL_COLS = ['makanan_pokok', 'lauk_pauk', 'sayur', 'buah', 'susu']
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = 'model/nutrikids_final.h5'
TEST_CSV = 'dataset/labels.csv'  # ganti dengan dataset uji terpisah
TEST_DIR = 'dataset/images'

# ============================================================
# MUAT MODEL & DATA UJI
# ============================================================
print("Memuat model dari", MODEL_PATH)
model = load_model(MODEL_PATH)

df_test = pd.read_csv(TEST_CSV)
df_test[LABEL_COLS] = df_test[LABEL_COLS].astype(float)

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_dataframe(
    dataframe=df_test, directory=TEST_DIR,
    x_col='filename', y_col=LABEL_COLS,
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='raw', shuffle=False   # shuffle=False agar urutan label konsisten
)

# ============================================================
# PREDIKSI & EVALUASI
# ============================================================
print("\nMemprediksi data uji...")
y_pred_prob = model.predict(test_gen)
y_pred = (y_pred_prob >= THRESHOLD).astype(int)

# Ambil label aktual
y_true = df_test[LABEL_COLS].astype(int).values

print(f"\nThreshold klasifikasi: {THRESHOLD}")
print(f"Jumlah data uji: {len(y_true)}")
print(f"Shape prediksi: {y_pred.shape}")

# ============================================================
# CLASSIFICATION REPORT PER LABEL (Bab 3.8.1)
# ============================================================
print("\n" + "=" * 60)
print("EVALUASI PER KELAS KOMPONEN MAKANAN")
print("=" * 60)

for i, label in enumerate(LABEL_COLS):
    print(f"\n{'─' * 40}")
    print(f"Kelas: {label.upper().replace('_', ' ')}")
    print(f"{'─' * 40}")

    cm = confusion_matrix(y_true[:, i], y_pred[:, i])
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(
        y_true[:, i], y_pred[:, i],
        target_names=['kurang', 'terdeteksi'],
        zero_division=0
    ))

# ============================================================
# VISUALISASI CONFUSION MATRIX
# ============================================================
fig, axes = plt.subplots(1, 5, figsize=(25, 4))
fig.suptitle('Confusion Matrix — NutriKids (Threshold ≥ 0.60)', fontsize=14)

for i, (label, ax) in enumerate(zip(LABEL_COLS, axes)):
    cm = confusion_matrix(y_true[:, i], y_pred[:, i])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['kurang', 'terdeteksi'],
                yticklabels=['kurang', 'terdeteksi'])
    ax.set_title(label.replace('_', ' ').title())
    ax.set_ylabel('Aktual')
    ax.set_xlabel('Prediksi')

plt.tight_layout()
plt.savefig('model/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n📊 Visualisasi tersimpan di model/confusion_matrices.png")

# ============================================================
# RINGKASAN METRIK KESELURUHAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN METRIK KESELURUHAN")
print("=" * 60)

from sklearn.metrics import accuracy_score, f1_score

for i, label in enumerate(LABEL_COLS):
    acc = accuracy_score(y_true[:, i], y_pred[:, i])
    f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
    print(f"  {label:20s}  Accuracy: {acc:.4f}  |  F1-Score: {f1:.4f}")

# Overall
overall_acc = accuracy_score(y_true.flatten(), y_pred.flatten())
overall_f1 = f1_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
print(f"\n  {'OVERALL':20s}  Accuracy: {overall_acc:.4f}  |  F1-Score: {overall_f1:.4f}")
