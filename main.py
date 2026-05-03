"""
=============================================================================
 PROYEK ATM — SEGMENTASI KANKER PARU-PARU PADA CITRA CT-SCAN
 Mata Kuliah: Computer Vision
 Metode: CLAHE + Otsu Thresholding + Morfologi (OpenCV)
 
 Referensi:
   - Jurnal 1: Husni & Adrial (2023), Jurnal Fisika Unand Vol.12 No.1
   - Jurnal 2: Fendriani dkk (2023), JoP Vol.8 No.2
 
 Catatan: Script ini dirancang untuk Google Colab.
          Jalankan setiap bagian (CELL) secara berurutan.
=============================================================================
"""

# ============================================================
# CELL 1 — MOUNT GOOGLE DRIVE & SETUP FOLDER
# ============================================================
# Fungsi: Menghubungkan Colab ke Google Drive agar hasil bisa
#         disimpan secara permanen (tidak hilang saat runtime putus).
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

import os
RESULTS_DIR = '/content/drive/MyDrive/ATM_KankerParu/results'
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Folder hasil: {RESULTS_DIR}")


# ============================================================
# CELL 2 — INSTALL LIBRARY TAMBAHAN
# ============================================================
# Fungsi: Menginstall kagglehub untuk download dataset dari Kaggle.
#         Library lain (cv2, numpy, matplotlib) sudah ada di Colab.
# ============================================================

# !pip install kagglehub -q


# ============================================================
# CELL 3 — IMPORT SEMUA LIBRARY
# ============================================================
# Fungsi masing-masing library:
#   cv2 (OpenCV)   : Pemrosesan gambar (CLAHE, threshold, morfologi)
#   numpy          : Operasi matematika pada array pixel
#   matplotlib     : Visualisasi gambar dan grafik
#   os, glob       : Navigasi file dan folder
#   pandas         : Membuat tabel ringkasan metrik
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import pandas as pd

print("OpenCV version:", cv2.__version__)
print("Setup selesai. Semua library berhasil di-import.")


# ============================================================
# CELL 4 — DOWNLOAD DAN EKSPLORASI DATASET
# ============================================================
# Fungsi: Mengunduh dataset CT-Scan dari Kaggle dan menghitung
#         jumlah gambar per kelas untuk dokumentasi.
#
# Dataset: mohamedhanyyy/chest-ctscan-images
# Berisi 4 kelas: adenocarcinoma, large.cell.carcinoma,
#                 normal, squamous.cell.carcinoma
# ============================================================

import kagglehub

path = kagglehub.dataset_download("mohamedhanyyy/chest-ctscan-images")
print("Dataset ada di:", path)

TRAIN_DIR = os.path.join(path, 'train')
TEST_DIR  = os.path.join(path, 'test')

print("\n=== Jumlah Gambar per Kelas (TEST) ===")
for folder in sorted(os.listdir(TEST_DIR)):
    folder_path = os.path.join(TEST_DIR, folder)
    if os.path.isdir(folder_path):
        jumlah = len(os.listdir(folder_path))
        print(f"  {folder}: {jumlah} gambar")

print("\n=== Jumlah Gambar per Kelas (TRAIN) ===")
for folder in sorted(os.listdir(TRAIN_DIR)):
    folder_path = os.path.join(TRAIN_DIR, folder)
    if os.path.isdir(folder_path):
        jumlah = len(os.listdir(folder_path))
        print(f"  {folder}: {jumlah} gambar")


# ============================================================
# CELL 5 — FUNGSI PREPROCESSING (CLAHE)
# ============================================================
# Fungsi: Membaca gambar, mengubah ke grayscale, me-resize,
#         lalu menerapkan CLAHE untuk meningkatkan kontras.
#
# CLAHE (Contrast Limited Adaptive Histogram Equalization):
#   - Meningkatkan kontras secara LOKAL (per region/grid)
#   - Lebih baik dari histogram equalization biasa karena
#     tidak membuat area tertentu terlalu terang/gelap
#   - clipLimit=2.0: membatasi amplifikasi kontras
#   - tileGridSize=(8,8): membagi gambar jadi grid 8x8
#
# Ini adalah MODIFIKASI dari jurnal referensi yang menggunakan
# perbaikan kontras manual (mapping linear 0.2-0.45 → 0-255).
# ============================================================

def preprocessing(img_path, img_size=256):
    """
    Baca gambar, ubah ke grayscale, resize, lalu terapkan CLAHE.

    Parameter:
        img_path (str): Path ke file gambar CT-Scan
        img_size (int): Ukuran output (default 256x256 pixel)

    Return:
        gray (ndarray)  : Gambar grayscale sebelum CLAHE
        hasil (ndarray) : Gambar grayscale setelah CLAHE
    """
    # Langkah 1: Baca gambar dari file
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Gambar tidak ditemukan: {img_path}")

    # Langkah 2: Ubah dari BGR (format OpenCV) ke Grayscale
    # CT-Scan pada dasarnya sudah grayscale, tapi file bisa disimpan
    # dalam format RGB/BGR sehingga perlu dikonversi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Langkah 3: Resize ke ukuran seragam
    # Semua gambar harus berukuran sama agar bisa dibandingkan
    gray = cv2.resize(gray, (img_size, img_size))

    # Langkah 4: Terapkan CLAHE (perbaikan kontras adaptif)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hasil = clahe.apply(gray)

    return gray, hasil


# ============================================================
# CELL 6 — FUNGSI SEGMENTASI (OTSU + MORFOLOGI)
# ============================================================
# Fungsi: Melakukan segmentasi gambar CT-Scan untuk memisahkan
#         area kanker dari jaringan sehat.
#
# Tahapan:
#   1. Otsu Thresholding — menentukan nilai threshold secara
#      OTOMATIS berdasarkan histogram gambar
#   2. Opening (Erosi → Dilasi) — menghilangkan noise kecil
#      di luar area kanker
#   3. Closing (Dilasi → Erosi) — menutup lubang kecil di
#      dalam area kanker
#
# Ini adalah MODIFIKASI UTAMA: jurnal hanya melakukan edge
# detection (menghasilkan garis tepi), sedangkan kita melakukan
# segmentasi (menghasilkan area terisi penuh).
# ============================================================

def segmentasi(img_clahe):
    """
    Segmentasi gambar menggunakan Otsu Thresholding + Morfologi.

    Parameter:
        img_clahe (ndarray): Gambar grayscale setelah CLAHE

    Return:
        mask_otsu (ndarray)   : Hasil Otsu (masih ada noise)
        mask_opening (ndarray): Setelah Opening (noise luar hilang)
        mask_closing (ndarray): Setelah Closing (lubang dalam tertutup) — FINAL
    """
    # Langkah 1: Otsu Thresholding
    # Parameter: 0 = threshold awal (diabaikan karena Otsu menghitung sendiri)
    #            255 = nilai pixel untuk area yang lolos threshold
    #            THRESH_BINARY + THRESH_OTSU = mode biner + algoritma Otsu
    _, mask_otsu = cv2.threshold(
        img_clahe, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Langkah 2: Buat kernel (elemen struktural) berbentuk elips 5x5
    # Elips dipilih karena bentuk tumor cenderung membulat
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Langkah 3: Opening = Erosi → Dilasi
    # Menghilangkan titik-titik putih kecil (noise) di luar area kanker
    mask_opening = cv2.morphologyEx(mask_otsu, cv2.MORPH_OPEN, kernel)

    # Langkah 4: Closing = Dilasi → Erosi
    # Menutup lubang-lubang hitam kecil di dalam area kanker
    mask_closing = cv2.morphologyEx(mask_opening, cv2.MORPH_CLOSE, kernel)

    return mask_otsu, mask_opening, mask_closing


# ============================================================
# CELL 7 — FUNGSI HITUNG MSE DAN PSNR
# ============================================================
# Fungsi: Menghitung metrik evaluasi yang SAMA dengan jurnal
#         referensi agar hasil bisa dibandingkan secara langsung.
#
# MSE (Mean Square Error):
#   - Rata-rata perbedaan kuadrat antara pixel asli dan hasil
#   - Semakin rendah = semakin mirip dengan asli
#
# PSNR (Peak Signal to Noise Ratio):
#   - Rasio kualitas sinyal dalam desibel (dB)
#   - > 30 dB = kualitas baik, < 10 dB = kualitas buruk
#
# Referensi pembanding:
#   Jurnal 1 (Canny terbaik): MSE=37278, PSNR=2.43 dB
#   Jurnal 2 (Median Filter): MSE=47, PSNR=31 dB
# ============================================================

def hitung_mse(img_asli, img_hasil):
    """
    Hitung MSE antara gambar asli dan gambar hasil segmentasi.

    Parameter:
        img_asli (ndarray) : Gambar grayscale asli (sebelum proses)
        img_hasil (ndarray): Gambar hasil segmentasi (mask biner)

    Return:
        float: Nilai MSE
    """
    asli  = img_asli.astype(np.float64)
    hasil = img_hasil.astype(np.float64)
    return np.mean((asli - hasil) ** 2)


def hitung_psnr(mse, max_pixel=255.0):
    """
    Hitung PSNR dari nilai MSE.

    Parameter:
        mse (float)      : Nilai MSE yang sudah dihitung
        max_pixel (float): Nilai pixel maksimum (255 untuk 8-bit)

    Return:
        float: Nilai PSNR dalam desibel (dB)
    """
    if mse == 0:
        return float('inf')
    return 10 * np.log10((max_pixel ** 2) / mse)


# ============================================================
# CELL 8 — FUNGSI PIPELINE LENGKAP (PROSES SATU GAMBAR)
# ============================================================
# Fungsi: Menjalankan seluruh pipeline dari awal sampai akhir
#         untuk satu gambar CT-Scan, termasuk visualisasi.
#
# Alur: Baca → Grayscale → CLAHE → Otsu → Opening → Closing
#       → Hitung MSE/PSNR → Tampilkan 5 panel visualisasi
# ============================================================

def proses_satu_gambar(img_path, simpan_path=None):
    """
    Pipeline lengkap untuk memproses satu gambar CT-Scan.

    Parameter:
        img_path (str)   : Path ke file gambar
        simpan_path (str): Path untuk menyimpan visualisasi (opsional)

    Return:
        dict: Berisi nama file dan nilai-nilai metrik
    """
    # === TAHAP 1: Preprocessing ===
    gray, clahe_img = preprocessing(img_path)

    # === TAHAP 2: Segmentasi ===
    mask_otsu, mask_opening, mask_closing = segmentasi(clahe_img)

    # === TAHAP 3: Hitung Metrik ===
    mse_otsu     = hitung_mse(gray, mask_otsu)
    psnr_otsu    = hitung_psnr(mse_otsu)
    mse_closing  = hitung_mse(gray, mask_closing)
    psnr_closing = hitung_psnr(mse_closing)

    # === TAHAP 4: Visualisasi 5 panel ===
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    judul  = ['Asli (Grayscale)', 'Setelah CLAHE', 'Otsu Threshold',
              'Setelah Opening', 'Setelah Closing (Final)']
    gambar = [gray, clahe_img, mask_otsu, mask_opening, mask_closing]

    for ax, img, title in zip(axes, gambar, judul):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    nama_file = os.path.basename(img_path)
    plt.suptitle(
        f'{nama_file}\n'
        f'Otsu → MSE: {mse_otsu:.2f}, PSNR: {psnr_otsu:.2f} dB | '
        f'Final → MSE: {mse_closing:.2f}, PSNR: {psnr_closing:.2f} dB',
        fontsize=9
    )
    plt.tight_layout()

    if simpan_path:
        plt.savefig(simpan_path, dpi=150, bbox_inches='tight')

    plt.show()

    return {
        'file'       : nama_file,
        'mse_otsu'   : round(mse_otsu,    2),
        'psnr_otsu'  : round(psnr_otsu,   2),
        'mse_final'  : round(mse_closing,  2),
        'psnr_final' : round(psnr_closing, 2)
    }


# ============================================================
# CELL 9 — JALANKAN UNTUK SEMUA KELAS
# ============================================================
# Fungsi: Mengambil 3 sampel gambar dari setiap kelas (4 kelas)
#         dan memproses masing-masing dengan pipeline lengkap.
#         Total: 4 kelas × 3 gambar = 12 gambar diproses.
# ============================================================

KELAS = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']
HASIL_SEMUA = []

for kelas in KELAS:
    folder = os.path.join(TEST_DIR, kelas)
    if not os.path.isdir(folder):
        print(f"  [SKIP] Folder tidak ditemukan: {folder}")
        continue

    # Ambil 3 gambar pertama dari tiap kelas sebagai sampel
    gambar_list = sorted(os.listdir(folder))[:3]

    print(f"\n{'='*50}")
    print(f"  KELAS: {kelas.upper()}")
    print(f"{'='*50}")

    for nama_gambar in gambar_list:
        img_path   = os.path.join(folder, nama_gambar)
        simpan     = os.path.join(RESULTS_DIR, f'{kelas}_{nama_gambar}.png')

        hasil = proses_satu_gambar(img_path, simpan_path=simpan)
        hasil['kelas'] = kelas
        HASIL_SEMUA.append(hasil)

        print(f"  {nama_gambar} → MSE: {hasil['mse_final']}, PSNR: {hasil['psnr_final']} dB")


# ============================================================
# CELL 10 — TABEL RINGKASAN METRIK
# ============================================================
# Fungsi: Menampilkan semua hasil dalam bentuk tabel dan
#         menghitung rata-rata per kelas serta keseluruhan.
#         Tabel disimpan sebagai CSV di Google Drive.
# ============================================================

df = pd.DataFrame(HASIL_SEMUA)

print("\n" + "="*70)
print("  TABEL HASIL SEGMENTASI")
print("="*70)
print(df[['kelas', 'file', 'mse_otsu', 'psnr_otsu', 'mse_final', 'psnr_final']].to_string(index=False))

print("\n" + "="*70)
print("  RATA-RATA PER KELAS")
print("="*70)
print(df.groupby('kelas')[['mse_final', 'psnr_final']].mean().round(2).to_string())

print(f"\n{'='*70}")
print(f"  RATA-RATA KESELURUHAN")
print(f"{'='*70}")
print(f"  MSE  rata-rata : {df['mse_final'].mean():.2f}")
print(f"  PSNR rata-rata : {df['psnr_final'].mean():.2f} dB")

# Simpan ke CSV
csv_path = os.path.join(RESULTS_DIR, 'metrics_table.csv')
df.to_csv(csv_path, index=False)
print(f"\n  Tabel tersimpan ke: {csv_path}")


# ============================================================
# CELL 11 — TABEL PERBANDINGAN FINAL DENGAN JURNAL (ATM)
# ============================================================
# Fungsi: Menampilkan perbandingan antara metode jurnal referensi
#         dan metode modifikasi kita untuk keperluan laporan.
# ============================================================

mse_avg  = df['mse_final'].mean()
psnr_avg = df['psnr_final'].mean()

print(f"""
{'='*75}
  TABEL PERBANDINGAN ATM (AMATI - TIRU - MODIFIKASI)
{'='*75}

+--------------------+-------------------------+-------------------------+---------------------------+
| Aspek              | Jurnal 1 (Husni 2023)   | Jurnal 2 (Fendriani)    | Penelitian Ini            |
+--------------------+-------------------------+-------------------------+---------------------------+
| Metode             | Robert, Sobel,          | Canny +                 | CLAHE + Otsu              |
|                    | Prewitt, Canny          | Mean/Median/Gaussian    | Thresholding + Morfologi  |
+--------------------+-------------------------+-------------------------+---------------------------+
| Preprocessing      | Perbaikan kontras       | Perbaikan kontras       | CLAHE (adaptif otomatis)  |
|                    | manual (0.2 - 0.45)     | manual                  |                           |
+--------------------+-------------------------+-------------------------+---------------------------+
| Hasil visual       | Garis tepi tipis        | Garis tepi + filter     | Area kanker terisi penuh  |
+--------------------+-------------------------+-------------------------+---------------------------+
| Dataset            | 5 pasien RS Unand       | 6 pasien RS Jambi       | Kaggle CT-Scan dataset    |
+--------------------+-------------------------+-------------------------+---------------------------+
| Tools              | MATLAB R2017b           | MATLAB R2015a           | Python + OpenCV (Colab)   |
+--------------------+-------------------------+-------------------------+---------------------------+
| MSE terbaik        | 37.278 (Canny)          | 47 (Median filter)      | {mse_avg:.2f}                |
+--------------------+-------------------------+-------------------------+---------------------------+
| PSNR terbaik       | 2,43 dB                 | 31 dB                   | {psnr_avg:.2f} dB              |
+--------------------+-------------------------+-------------------------+---------------------------+

Catatan:
  - Jurnal 1 & 2 menggunakan Edge Detection (menghasilkan garis tepi)
  - Penelitian ini menggunakan Segmentasi (menghasilkan area terisi)
  - MSE tinggi pada jurnal 1 karena edge detection sangat mengubah gambar
""")

print("="*75)
print("  PROYEK ATM SELESAI!")
print("  Semua hasil tersimpan di Google Drive: ATM_KankerParu/results/")
print("="*75)
