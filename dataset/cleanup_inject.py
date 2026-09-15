"""
Membersihkan data injeksi lama (BENTO_INJECT) dari dataset
agar dataset kembali bersih sebelum augmentasi yang benar.
"""
import os
import pandas as pd

LABELS_CSV = 'dataset/labels.csv'
DATASET_DIR = 'dataset/images'

df = pd.read_csv(LABELS_CSV)
print(f"Sebelum: {len(df)} entri")

# Hapus baris BENTO_INJECT
mask_inject = df['filename'].str.contains('BENTO_INJECT', na=False)
inject_files = df[mask_inject]['filename'].tolist()

# Hapus file fisik
deleted = 0
for fname in inject_files:
    fpath = os.path.join(DATASET_DIR, fname)
    if os.path.exists(fpath):
        os.remove(fpath)
        deleted += 1

# Hapus juga gambar sintetis lama jika ada
mask_synth = df['filename'].str.startswith('synth_bento', na=False)
mask_aug = df['filename'].str.startswith('aug_', na=False)
synth_files = df[mask_synth]['filename'].tolist() + df[mask_aug]['filename'].tolist()
for fname in synth_files:
    fpath = os.path.join(DATASET_DIR, fname)
    if os.path.exists(fpath):
        os.remove(fpath)
        deleted += 1

# Simpan CSV bersih
df_clean = df[~mask_inject & ~mask_synth & ~mask_aug]
df_clean.to_csv(LABELS_CSV, index=False)

print(f"Sesudah: {len(df_clean)} entri")
print(f"Dihapus: {len(inject_files) + len(synth_files)} entri CSV, {deleted} file gambar")
print("Dataset bersih!")
