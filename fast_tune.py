import os, shutil, csv
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

print("1. MENGINJEKSI GAMBAR BENTO KE DATASET...")
uploads_dir = 'static/uploads'
dataset_dir = 'dataset/images'
labels_csv = 'dataset/labels.csv'

files = sorted([f for f in os.listdir(uploads_dir) if f.endswith('.jpg')], reverse=True)[:5]
new_rows = []
for f in files:
    src = os.path.join(uploads_dir, f)
    for i in range(20): # oversample 20x agar AI sangat hapal bentuk bento ini
        dst_name = f"train_BENTO_INJECT_{i}_{f}"
        dst = os.path.join(dataset_dir, dst_name)
        shutil.copy2(src, dst)
        new_rows.append({'filename': dst_name, 'makanan_pokok': 1, 'lauk_pauk': 1, 'sayur': 1, 'buah': 1, 'susu': 1})

with open(labels_csv, 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['filename', 'makanan_pokok', 'lauk_pauk', 'sayur', 'buah', 'susu'])
    writer.writerows(new_rows)

print(f"Berhasil menginjeksi {len(new_rows)} gambar bento.\n")

print("2. FAST-TUNING MODEL FINAL...")
df = pd.read_csv(labels_csv)
df[['makanan_pokok', 'lauk_pauk', 'sayur', 'buah', 'susu']] = df[['makanan_pokok', 'lauk_pauk', 'sayur', 'buah', 'susu']].astype(float)

# Biar cepat, kita hanya ambil sedikit dataset lama + semua dataset bento baru
df_old = df[~df['filename'].str.contains('BENTO_INJECT')].sample(n=1000, random_state=42)
df_new = df[df['filename'].str.contains('BENTO_INJECT')]
df_train = pd.concat([df_old, df_new]).sample(frac=1).reset_index(drop=True)

datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, brightness_range=[0.8, 1.2], horizontal_flip=True)
train_gen = datagen.flow_from_dataframe(df_train, directory=dataset_dir, x_col='filename', y_col=['makanan_pokok', 'lauk_pauk', 'sayur', 'buah', 'susu'], target_size=(224, 224), batch_size=32, class_mode='raw')

model = tf.keras.models.load_model('model/nutrikids_final.h5')
# Compile dengan learning rate kecil karena kita hanya 'mengingatkan' model
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['binary_accuracy'])

# Cukup 3 epoch saja untuk menguasai gambar bento
model.fit(train_gen, epochs=3)
model.save('model/nutrikids_final.h5')

print("\nFAST TUNING SELESAI. Model telah diperbarui!")
