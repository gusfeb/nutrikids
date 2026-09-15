import os, cv2, csv, requests, numpy as np, time
from duckduckgo_search import DDGS

IMG_DIR = os.path.join(os.path.dirname(__file__), 'images')
LABELS_FILE = os.path.join(os.path.dirname(__file__), 'labels.csv')
IMG_SIZE = (224, 224)

FRUIT_QUERIES = [
    'buah nanas segar', 'buah stroberi piring', 'buah naga merah',
    'buah alpukat potong', 'buah kiwi segar', 'buah pir manis',
    'buah rambutan', 'buah duku manis', 'buah manggis', 'buah kelengkeng'
]

os.makedirs(IMG_DIR, exist_ok=True)
fields = ['filename', 'makanan_pokok', 'lauk_pauk', 'sayur', 'buah', 'susu']
new_rows = []
total = 0

ddgs = DDGS()
for q in FRUIT_QUERIES:
    print(f"Mencari {q} via DDGS...")
    try:
        results = list(ddgs.images(q, max_results=15))
        urls = [r['image'] for r in results if 'image' in r]
    except Exception as e:
        print(f"Error: {e}")
        urls = []
        
    for url in urls:
        try:
            res = requests.get(url, timeout=5)
            img = cv2.imdecode(np.asarray(bytearray(res.content), dtype="uint8"), cv2.IMREAD_COLOR)
            if img is not None:
                fname = f"train_BuahX_{q.replace(' ','_')}_{total:04d}.jpg"
                cv2.imwrite(os.path.join(IMG_DIR, fname), cv2.resize(img, IMG_SIZE))
                new_rows.append({'filename': fname, 'makanan_pokok': 0, 'lauk_pauk': 0, 'sayur': 0, 'buah': 1, 'susu': 0})
                total += 1
        except:
            pass
    time.sleep(2)

if new_rows:
    with open(LABELS_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writerows(new_rows)
print(f"Selesai! Berhasil mengunduh {total} gambar buah tambahan.")
