import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

print("Memuat model final...")
model = tf.keras.models.load_model('model/nutrikids_final.h5')
uploads_dir = 'static/uploads'

# Ambil 4 file terakhir
files = sorted([f for f in os.listdir(uploads_dir) if f.endswith('.jpg')])[-4:]

print("\nHasil Prediksi Model pada 4 Foto Terakhir:")
for f in files:
    img_path = os.path.join(uploads_dir, f)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    preds = model.predict(x, verbose=0)[0]
    
    print(f"\nFile: {f}")
    print(f"Pokok: {preds[0]:.2%} | Lauk: {preds[1]:.2%} | Sayur: {preds[2]:.2%} | Buah: {preds[3]:.2%} | Susu: {preds[4]:.2%}")
