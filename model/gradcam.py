"""
NutriKids — Grad-CAM Visualization Module
==========================================
Menggunakan Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017)
untuk memvisualisasikan area mana dalam gambar yang paling diperhatikan model
saat mendeteksi setiap kategori makanan.

Referensi Akademis:
  Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. and Batra, D.
  "Grad-CAM: Visual Explanations from Deep Neural Networks via Gradient-based Localization"
  ICCV 2017
"""

import numpy as np
import cv2
import tensorflow as tf


def generate_gradcam(model, img_array, class_index, last_conv_layer_name=None):
    """
    Generate Grad-CAM heatmap untuk satu class index tertentu.
    
    Args:
        model: Keras model (MobileNetV2 + custom top)
        img_array: numpy array shape (1, 224, 224, 3), sudah dinormalisasi
        class_index: index label (0-4)
        last_conv_layer_name: nama layer konvolusi terakhir (auto-detect jika None)
    
    Returns:
        heatmap: numpy array (224, 224) dengan nilai 0-1
        peak_point: tuple (x, y) titik puncak aktivasi
    """
    # Auto-detect last conv layer dari MobileNetV2
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
            # Untuk model yang menggunakan MobileNetV2 sebagai functional submodel
            if hasattr(layer, 'layers'):
                for sub_layer in reversed(layer.layers):
                    if isinstance(sub_layer, tf.keras.layers.Conv2D):
                        last_conv_layer_name = layer.name  # use the parent layer name
                        break
                if last_conv_layer_name:
                    break
    
    if last_conv_layer_name is None:
        # Fallback: cari layer dengan 'conv' di namanya
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv_layer_name = layer.name
                break

    # Buat sub-model: input -> [last_conv_output, predictions]
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
        # Jika layer adalah sebuah Model (misal mobilenetv2), ambil output terakhirnya
        if hasattr(last_conv_layer, 'output'):
            conv_output = last_conv_layer.output
        else:
            conv_output = last_conv_layer
            
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[conv_output, model.output]
        )
    except Exception:
        # Jika gagal, coba cara alternatif untuk MobileNetV2 wrapper
        # Ambil layer 'out_relu' dari dalam mobilenetv2
        base = None
        for layer in model.layers:
            if hasattr(layer, 'layers') and len(layer.layers) > 10:
                base = layer
                break
        
        if base is None:
            # Return blank heatmap
            return np.zeros((224, 224), dtype=np.float32), (112, 112)
        
        # Cari 'out_relu' atau conv terakhir di dalam base model
        target_layer = None
        for layer in reversed(base.layers):
            if 'out_relu' in layer.name or ('conv' in layer.name and isinstance(layer, tf.keras.layers.Conv2D)):
                target_layer = layer
                break
        
        if target_layer is None:
            return np.zeros((224, 224), dtype=np.float32), (112, 112)
        
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[base.get_layer(target_layer.name).output, model.output]
        )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        target_class = predictions[:, class_index]

    grads = tape.gradient(target_class, conv_outputs)
    
    if grads is None:
        return np.zeros((224, 224), dtype=np.float32), (112, 112)

    # Global Average Pooling pada gradien
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted combination
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU dan normalisasi
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize ke ukuran gambar
    heatmap = cv2.resize(heatmap, (224, 224))

    # Cari titik puncak aktivasi
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

    return heatmap, (int(peak_x), int(peak_y))


def create_gradcam_overlay(original_img, model, img_array, labels, scores, threshold=0.60):
    """
    Buat gambar overlay dengan Grad-CAM heatmap dan label garis untuk setiap
    kategori yang terdeteksi.
    
    Args:
        original_img: gambar asli (BGR, ukuran asli)
        model: Keras model
        img_array: numpy array (1, 224, 224, 3) sudah dinormalisasi
        labels: list nama label ['makanan_pokok', 'lauk_pauk', ...]
        scores: list skor prediksi [0.95, 0.82, ...]
        threshold: ambang batas deteksi
    
    Returns:
        annotated_img: gambar BGR dengan overlay Grad-CAM dan label
    """
    h, w = original_img.shape[:2]
    annotated = original_img.copy()
    
    # Warna untuk setiap kategori
    COLORS = {
        'makanan_pokok': (188, 71, 171),   # ungu (BGR)
        'lauk_pauk':     (255, 229, 0),    # cyan (BGR)
        'sayur':         (129, 64, 255),   # pink (BGR)
        'buah':          (0, 171, 255),    # kuning/oranye (BGR)
        'susu':          (53, 107, 255),   # oranye/coral (BGR)
    }
    
    LABEL_NAMES = {
        'makanan_pokok': 'Karbohidrat',
        'lauk_pauk':     'Protein',
        'sayur':         'Sayuran',
        'buah':          'Buah',
        'susu':          'Susu',
    }
    
    detected_items = []
    
    for i, (label, score) in enumerate(zip(labels, scores)):
        if score < threshold:
            continue
        
        # Generate Grad-CAM heatmap
        heatmap, peak_224 = generate_gradcam(model, img_array, i)
        
        # Skala titik puncak ke ukuran gambar asli
        peak_x = int(peak_224[0] * w / 224)
        peak_y = int(peak_224[1] * h / 224)
        
        # Pastikan titik dalam batas gambar
        peak_x = max(20, min(peak_x, w - 20))
        peak_y = max(20, min(peak_y, h - 20))
        
        color = COLORS.get(label, (255, 255, 255))
        name = LABEL_NAMES.get(label, label)
        persen = int(score * 100)
        
        detected_items.append({
            'label': name,
            'peak': (peak_x, peak_y),
            'color': color,
            'score': persen,
            'heatmap': heatmap
        })
    
    # Overlay heatmap gabungan (semi-transparan)
    if detected_items:
        combined_heatmap = np.zeros((224, 224, 3), dtype=np.float32)
        for item in detected_items:
            color_norm = np.array(item['color'], dtype=np.float32) / 255.0
            hm = item['heatmap']
            for c in range(3):
                combined_heatmap[:, :, c] += hm * color_norm[c]
        
        combined_heatmap = np.clip(combined_heatmap, 0, 1)
        combined_heatmap_resized = cv2.resize(combined_heatmap, (w, h))
        
        # Blend dengan gambar asli (alpha = 0.3)
        overlay = (combined_heatmap_resized * 255).astype(np.uint8)
        annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)
    
    # Gambar label garis untuk setiap komponen terdeteksi
    used_label_positions = []
    
    for idx, item in enumerate(detected_items):
        peak = item['peak']
        color = item['color']
        name = item['label']
        persen = item['score']
        
        # Tentukan posisi label (di sisi kanan atau kiri gambar)
        # Agar tidak saling tumpuk
        margin = 15
        label_text = f"{name} {persen}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.45, min(0.7, w / 800))
        thickness = max(1, int(w / 500))
        
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        # Posisi label di sisi kanan
        label_x = w - text_w - margin - 10
        label_y = margin + (idx * (text_h + 30))
        
        # Pastikan tidak keluar batas
        label_y = max(text_h + 5, min(label_y, h - 10))
        
        # Titik ujung garis di label
        line_end_x = label_x - 5
        line_end_y = label_y - text_h // 2
        
        # Gambar garis dari titik puncak ke label
        # Garis dengan 2 segmen (horizontal lalu vertikal) agar rapi
        mid_x = (peak[0] + line_end_x) // 2
        
        # Garis utama (tebal, berwarna)
        cv2.line(annotated, peak, (mid_x, peak[1]), color, thickness + 1, cv2.LINE_AA)
        cv2.line(annotated, (mid_x, peak[1]), (mid_x, line_end_y), color, thickness + 1, cv2.LINE_AA)
        cv2.line(annotated, (mid_x, line_end_y), (line_end_x, line_end_y), color, thickness + 1, cv2.LINE_AA)
        
        # Titik bulat di peak
        cv2.circle(annotated, peak, max(4, int(w / 120)), color, -1, cv2.LINE_AA)
        cv2.circle(annotated, peak, max(6, int(w / 90)), color, thickness, cv2.LINE_AA)
        
        # Background label (rounded rectangle)
        bg_x1 = label_x - 8
        bg_y1 = label_y - text_h - 8
        bg_x2 = label_x + text_w + 8
        bg_y2 = label_y + 8
        
        # Semi-transparent background
        sub_img = annotated[max(0, bg_y1):min(h, bg_y2), max(0, bg_x1):min(w, bg_x2)]
        if sub_img.size > 0:
            dark_bg = np.full_like(sub_img, (30, 30, 30))
            blended = cv2.addWeighted(sub_img, 0.3, dark_bg, 0.7, 0)
            annotated[max(0, bg_y1):min(h, bg_y2), max(0, bg_x1):min(w, bg_x2)] = blended
        
        # Border label
        cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), color, thickness, cv2.LINE_AA)
        
        # Text label
        cv2.putText(annotated, label_text, (label_x, label_y),
                    font, font_scale, (255, 255, 255), thickness + 1, cv2.LINE_AA)
        cv2.putText(annotated, label_text, (label_x, label_y),
                    font, font_scale, color, thickness, cv2.LINE_AA)
    
    return annotated
