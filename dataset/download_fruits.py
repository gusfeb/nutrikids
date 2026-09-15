import os
import cv2
import csv
import requests
from bs4 import BeautifulSoup
import numpy as np

IMG_DIR = os.path.join(os.path.dirname(__file__), 'images')
LABELS_FILE = os.path.join(os.path.dirname(__file__), 'labels.csv')
IMG_SIZE = (224, 224)

# Konfigurasi Kategori dan Kata Kunci Pencariannya
SEARCH_CONFIG = {
    'buah': ['buah apel', 'buah pisang', 'buah jeruk', 'buah mangga', 'buah semangka', 'buah pepaya', 'buah melon', 'buah anggur piring'],
    'makanan_pokok': ['nasi putih piring', 'nasi kuning bekal', 'nasi goreng porsi anak', 'nasi merah bekal', 'roti gandum lembaran', 'kentang rebus piring', 'mie goreng bekal'],
    'sayur': ['sayur bayam mangkuk', 'sayur sop sayuran', 'tumis kangkung', 'capcay sayur segar', 'tumis buncis', 'sayur asem', 'brokoli rebus'],
    'susu': ['susu kotak uht', 'susu botol anak', 'susu putih gelas', 'susu coklat kotak', 'susu dancow gelas', 'susu bear brand'],
    'lauk_pauk': ['tempe orek piring', 'telur balado', 'ayam goreng paha', 'tahu goreng', 'ikan goreng piring', 'telur dadar piring', 'ayam kecap porsi']
}

TARGET_PER_QUERY = 50 # Diperbanyak menjadi 50 gambar per kata kunci

def search_images_ddg(query, max_results=50):
    url = "https://html.duckduckgo.com/html/?q=" + requests.utils.quote(query + " filetype:jpg")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        
        image_urls = []
        for img_tag in soup.find_all('img', class_='tile--img__img'):
            src = img_tag.get('src')
            if src and src.startswith('//'):
                image_urls.append('https:' + src)
        return image_urls[:max_results]
    except Exception as e:
        print(f"Gagal mencari '{query}': {e}")
        return []

def download_extra_data():
    os.makedirs(IMG_DIR, exist_ok=True)
    
    existing_labels = []
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_labels = list(reader)
    
    fields = ['filename', 'makanan_pokok', 'lauk_pauk', 'sayur', 'buah', 'susu']
    total_downloaded = 0
    new_rows = []
    
    print("==================================================")
    print("Mulai mencari dan mengunduh gambar tambahan...")
    print("==================================================")
    
    try:
        from duckduckgo_search import DDGS
        use_ddgs = True
    except ImportError:
        use_ddgs = False

    for category, queries in SEARCH_CONFIG.items():
        print(f"\n---> MEMPROSES KATEGORI: {category.upper()} <---")
        
        for query in queries:
            print(f"\nMencari: '{query}'")
            
            image_urls = []
            if use_ddgs:
                try:
                    with DDGS() as ddgs:
                        results = list(ddgs.images(query, max_results=TARGET_PER_QUERY))
                        image_urls = [r['image'] for r in results if 'image' in r]
                except Exception as e:
                    print(f"DDGS error: {e}")
            
            # Jika DDGS gagal (misal 403 Ratelimit) atau tidak mendapat hasil, gunakan fallback scraping
            if not image_urls:
                print(f"Menggunakan metode fallback untuk '{query}'...")
                image_urls = search_images_ddg(query, TARGET_PER_QUERY)
                
            success_count = 0
            for i, url in enumerate(image_urls):
                try:
                    response = requests.get(url, timeout=5)
                    response.raise_for_status()
                    
                    image_data = np.asarray(bytearray(response.content), dtype="uint8")
                    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    
                    img_resized = cv2.resize(img, IMG_SIZE)
                    
                    safe_query_name = query.replace(' ', '_')
                    filename = f"train_Ekstra_{category}_{safe_query_name}_{total_downloaded:04d}.jpg"
                    filepath = os.path.join(IMG_DIR, filename)
                    
                    cv2.imwrite(filepath, img_resized)
                    
                    # Dinamis mengubah label sesuai kategori
                    row = {
                        'filename': filename,
                        'makanan_pokok': 1 if category == 'makanan_pokok' else 0,
                        'lauk_pauk': 1 if category == 'lauk_pauk' else 0,
                        'sayur': 1 if category == 'sayur' else 0,
                        'buah': 1 if category == 'buah' else 0,
                        'susu': 1 if category == 'susu' else 0
                    }
                    new_rows.append(row)
                    
                    success_count += 1
                    total_downloaded += 1
                    
                except Exception:
                    pass
                    
            print(f"--> Berhasil mengunduh {success_count} gambar untuk '{query}'.")

    if new_rows:
        with open(LABELS_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not existing_labels:
                writer.writeheader()
            writer.writerows(new_rows)
            
        print("\n==================================================")
        print(f"SELESAI! Berhasil mengunduh total {total_downloaded} gambar tambahan.")
        print("Dataset berhasil diperbarui dan labels.csv telah ditambahkan.")
        print("==================================================")
    else:
        print("\n[!] Tidak ada gambar yang berhasil diunduh.")

if __name__ == "__main__":
    download_extra_data()
