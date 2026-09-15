"""
NutriKids — Training Pipeline
Klasifikasi Multi-Label Komponen Makanan Siswa dengan MobileNetV2
Berdasarkan Bab 2.4, 2.5, 3.4, 3.5.1 Proposal Seminar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import json
import os

# ============================================================
# KONFIGURASI
# ============================================================
LABEL_COLS = ['makanan_pokok', 'lauk_pauk', 'sayur', 'buah', 'susu']
IMG_SIZE = (224, 224)   # standar input MobileNetV2 (Bab 3.4.2)
BATCH_SIZE = 32
NUM_CLASSES = 5
DATASET_DIR = 'dataset/images'
LABELS_CSV = 'dataset/labels.csv'

# ============================================================
# TAHAP 2 — PRA-PEMROSESAN & AUGMENTASI DATA (Bab 3.4.2 & 3.4.3)
# ============================================================
print("=" * 60)
print("TAHAP 2: Memuat & Augmentasi Dataset")
print("=" * 60)

df = pd.read_csv(LABELS_CSV)
# Gunakan tipe data float karena kita menggunakan class_mode='raw' (multi-label)
df[LABEL_COLS] = df[LABEL_COLS].astype(float)

# --- BALANCING DATASET ---
# Untuk mengatasi Class Imbalance (Lauk Pauk 6000 vs Buah 200)
# Kita membatasi maksimal sampel per kategori (undersampling).
print("\nMelakukan balancing dataset untuk mencegah bias...")
balanced_dfs = []
for col in LABEL_COLS:
    df_col = df[df[col] == 1]
    if len(df_col) > 600:
        df_col = df_col.sample(n=600, random_state=42)
    balanced_dfs.append(df_col)

# Gabungkan dan hapus duplikat (karena gambar bisa punya multi-label)
df = pd.concat(balanced_dfs).drop_duplicates(subset=['filename'])
df = df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle

print(f"Total gambar dalam dataset (setelah diseimbangkan): {len(df)}")
print(f"Distribusi label baru:\n{df[LABEL_COLS].astype(int).sum()}\n")

datagen = ImageDataGenerator(
    rescale=1./255,              # normalisasi (Bab 3.4.3)
    rotation_range=20,           # augmentasi (Bab 3.2.1)
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    validation_split=0.2         # 80% training, 20% validasi
)

train_gen = datagen.flow_from_dataframe(
    dataframe=df, directory=DATASET_DIR,
    x_col='filename', y_col=LABEL_COLS,
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='raw', subset='training'
)
# class_mode='raw' karena satu gambar bisa punya lebih dari satu label
# aktif sekaligus — multi-label, sesuai Bab 2.3.4

val_gen = datagen.flow_from_dataframe(
    dataframe=df, directory=DATASET_DIR,
    x_col='filename', y_col=LABEL_COLS,
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='raw', subset='validation'
)

# ============================================================
# TAHAP 3 — ARSITEKTUR MODEL (Bab 3.5.1)
# ============================================================
print("=" * 60)
print("TAHAP 3: Membangun Arsitektur MobileNetV2 + Custom Top")
print("=" * 60)

# 1) Base model — pretrained ImageNet, top layer dibuang
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False   # freeze dulu untuk fase Transfer Learning

# 2) Custom top layers (Bab 3.5.1 poin 2)
x = base_model.output
x = GlobalAveragePooling2D()(x)          # reduksi dimensi spasial
x = Dense(128, activation='relu')(x)     # dense layer
output = Dense(NUM_CLASSES, activation='sigmoid')(x)  # 5 node, sigmoid independen

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),   # LR fase awal (Bab 3.5.1 poin 3)
    loss='binary_crossentropy',           # sesuai Bab 3.5.1
    metrics=['binary_accuracy']
)

print("\n--- Model Summary ---")
model.summary()

# ============================================================
# TAHAP 4 — TRAINING FASE TRANSFER LEARNING
# ============================================================
print("\n" + "=" * 60)
print("TAHAP 4: Training Fase Transfer Learning (base frozen)")
print("=" * 60)

class TrainingMonitorCallback(Callback):
    def __init__(self, phase="Transfer Learning"):
        super().__init__()
        self.phase = phase
        self.status_file = 'model/training_status.json'
        # Reset file
        with open(self.status_file, 'w') as f:
            json.dump({'status': 'Menyiapkan...', 'phase': phase, 'epoch': 0, 'logs': {}}, f)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Convert float32 to standard float for JSON serialization
        clean_logs = {k: float(v) for k, v in logs.items()}
        data = {
            'status': 'Training',
            'phase': self.phase,
            'epoch': epoch + 1,
            'logs': clean_logs
        }
        with open(self.status_file, 'w') as f:
            json.dump(data, f)
            
    def on_train_end(self, logs=None):
        data = {
            'status': 'Selesai',
            'phase': self.phase,
            'epoch': 'Selesai',
            'logs': {}
        }
        with open(self.status_file, 'w') as f:
            json.dump(data, f)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('model/nutrikids_best.h5', save_best_only=True),
    TrainingMonitorCallback(phase="Transfer Learning")
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=100,          # maks 100 epoch + early stopping (Bab 3.5.1)
    callbacks=callbacks
)

# ============================================================
# TAHAP 5 — FINE-TUNING (Bab 2.5)
# ============================================================
print("\n" + "=" * 60)
print("TAHAP 5: Fine-Tuning (membuka lapisan atas MobileNetV2)")
print("=" * 60)

# Buka beberapa lapisan atas MobileNetV2
base_model.trainable = True
FINE_TUNE_AT = 100   # bekukan lapisan sebelum index ini

for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),   # LR diperkecil (Bab 2.5 & 3.5.1)
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

callbacks_fine = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('model/nutrikids_best.h5', save_best_only=True),
    TrainingMonitorCallback(phase="Fine-Tuning")
]

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=callbacks_fine
)

# Simpan model final
model.save('model/nutrikids_final.h5')
print("\n[OK] Model tersimpan di model/nutrikids_final.h5")

# ============================================================
# PLOT HASIL TRAINING
# ============================================================
def plot_training(history, history_fine):
    """Visualisasi loss & accuracy selama training"""
    # Gabungkan history
    acc = history.history['binary_accuracy'] + history_fine.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy'] + history_fine.history['val_binary_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot accuracy
    ax1.plot(acc, label='Training Accuracy')
    ax1.plot(val_acc, label='Validation Accuracy')
    ax1.axvline(len(history.history['binary_accuracy']), linestyle='--', color='gray',
                label='Mulai Fine-Tuning')
    ax1.set_title('Binary Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    # Plot loss
    ax2.plot(loss, label='Training Loss')
    ax2.plot(val_loss, label='Validation Loss')
    ax2.axvline(len(history.history['loss']), linestyle='--', color='gray',
                label='Mulai Fine-Tuning')
    ax2.set_title('Binary Crossentropy Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('model/training_history.png', dpi=150)
    plt.show()
    print("📊 Grafik tersimpan di model/training_history.png")

plot_training(history, history_fine)
