# 🤖 PANDUAN UNTUK ANTIGRAVITY — ATM PROJECT
## Mata Kuliah: Computer Vision
## Topik: Segmentasi Kanker Paru-Paru pada Citra CT-Scan

> Baca dokumen ini dari awal sebelum mulai coding.
> Semua metode di sini adalah Computer Vision klasik — tidak ada Deep Learning.

---

# BAGIAN 1 — BASIC KNOWLEDGE

---

## 1.1 Konteks: Apa yang Sudah Dilakukan Jurnal Referensi?

Ada dua jurnal yang menjadi acuan:

### Jurnal 1 — Husni & Adrial (2023)
- **Metode:** Edge Detection → Robert, Sobel, Prewitt, Canny
- **Data:** 5 pasien CT Simulator, RS Universitas Andalas
- **Tools:** MATLAB R2017b
- **Hasil terbaik:** Metode Canny → MSE rata-rata 37.278, PSNR 2,43 dB
- **Kesimpulan:** Hanya Canny yang bisa menunjukkan batas kanker, tapi masih banyak artifact

### Jurnal 2 — Fendriani dkk (2023)
- **Metode:** Canny + variasi Filter → Mean, Median, Gaussian
- **Data:** 6 pasien CT-Scan, RS Raden Mattaher Jambi
- **Tools:** MATLAB R2015a
- **Hasil terbaik:** Filter Median → MSE 47, PSNR 31 dB
- **Kesimpulan:** Filter Median paling efektif mengurangi noise pada hasil Canny

### Kelemahan Kedua Jurnal
- Edge detection hanya menemukan **garis tepi** — tidak memisahkan area kanker secara utuh
- Hasilnya berupa gambar hitam-putih dengan banyak garis, bukan area kanker yang jelas
- Tidak ada informasi tentang **ukuran** atau **bentuk** area kanker

---

## 1.2 Apa Bedanya Edge Detection vs Segmentasi?

Ini adalah inti dari **MODIFIKASI** yang kita lakukan.

| Aspek | Edge Detection (Jurnal) | Segmentasi (Kita) |
|-------|------------------------|-------------------|
| Hasil visual | Garis tepi tipis | Area kanker terisi penuh |
| Informasi | Hanya batas/tepi | Bentuk & ukuran area kanker |
| Analogi | Menggambar garis pinggiran objek | Mewarnai seluruh objek |
| Metode | Robert, Sobel, Prewitt, Canny | Thresholding + Morfologi |

**Analogi sederhana:**
- Edge Detection = seperti menggambar outline pensil di sekitar benda
- Segmentasi = seperti mengecat benda tersebut dengan warna solid

---

## 1.3 Apa Itu Segmentasi Citra?

Segmentasi adalah proses **memisahkan objek yang diminati dari latar belakang** dalam sebuah gambar.

Pada kasus CT-Scan kanker paru-paru:
- Kita ingin memisahkan **area tumor/kanker** dari **jaringan sehat dan latar belakang**
- Hasilnya berupa **mask** (gambar biner hitam-putih) di mana putih = kanker, hitam = bukan kanker

---

## 1.4 Metode yang Kita Gunakan

### A. Otsu Thresholding

Thresholding adalah teknik paling dasar dalam segmentasi. Cara kerjanya:
- Tentukan nilai ambang batas (threshold) T
- Pixel dengan nilai > T → dijadikan putih (objek)
- Pixel dengan nilai ≤ T → dijadikan hitam (latar belakang)

**Masalah thresholding biasa:** Nilai T harus ditentukan manual — hasilnya berbeda-beda tergantung gambar.

**Solusi: Otsu Thresholding**
Otsu secara **otomatis** mencari nilai T terbaik berdasarkan histogram gambar.
Algoritma Otsu mencari nilai threshold yang memaksimalkan perbedaan antara dua kelompok pixel (kanker vs bukan kanker).

```
Nilai T optimal = nilai yang membuat variansi antar-kelompok paling besar
```

Di OpenCV, ini cukup satu baris:
```python
_, hasil = cv2.threshold(gambar_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

### B. CLAHE (Contrast Limited Adaptive Histogram Equalization)

Sebelum segmentasi, kualitas gambar perlu ditingkatkan terlebih dahulu.

**Histogram Equalization biasa:** Meratakan distribusi kecerahan seluruh gambar sekaligus.
Masalahnya: bisa membuat area tertentu terlalu terang atau terlalu gelap.

**CLAHE:** Melakukan histogram equalization secara **lokal** (dibagi dalam grid kecil),
bukan satu gambar sekaligus. Hasilnya lebih natural dan detail lokal lebih terjaga.

Ini adalah **pengganti perbaikan kontras manual** yang dilakukan jurnal referensi
(jurnal melakukan mapping linear 0.2–0.45 → 0–255, kita pakai CLAHE yang otomatis dan adaptif).

```python
clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
hasil  = clahe.apply(gambar_gray)
```

### C. Operasi Morfologi

Setelah thresholding, hasil segmentasi biasanya masih "kotor" — ada lubang kecil di dalam area kanker atau noise kecil di luar. Morfologi digunakan untuk membersihkan ini.

**Opening (Erosi → Dilasi)**
- Menghilangkan noise kecil di luar objek utama
- Analogi: "mengikis" sedikit pinggiran lalu "menebalkan" kembali

**Closing (Dilasi → Erosi)**
- Menutup lubang kecil di dalam objek
- Analogi: "mengisi" celah-celah kecil di dalam area kanker

```python
kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)  # buang noise luar
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # tutup lubang dalam
```

---

## 1.5 Metrik Evaluasi (Sama Seperti Jurnal Referensi)

Kita pakai metrik yang **sama** dengan jurnal agar bisa dibandingkan langsung.

### MSE (Mean Square Error)
Mengukur rata-rata perbedaan kuadrat antara gambar asli dan gambar hasil olahan.

```
MSE = (1 / M×N) × Σ [I(i,j) - K(i,j)]²

I(i,j) = nilai pixel gambar asli
K(i,j) = nilai pixel gambar hasil segmentasi
M, N   = ukuran gambar (baris × kolom)
```

- MSE rendah → gambar hasil mirip gambar asli (sedikit berubah)
- MSE tinggi → gambar hasil sangat berbeda dari asli (banyak berubah)

### PSNR (Peak Signal to Noise Ratio)
Mengukur kualitas gambar dalam satuan desibel (dB).

```
PSNR = 10 × log₁₀ (255² / MSE)

Nilai 255 = nilai pixel maksimum pada gambar 8-bit
```

- PSNR > 30 dB → kualitas baik (sedikit noise)
- PSNR < 10 dB → kualitas buruk (banyak noise)

**Catatan penting dari jurnal:**
Jurnal 1 memiliki PSNR sangat rendah (~2 dB) karena edge detection sangat mengubah gambar.
Jurnal 2 dengan filter median mencapai PSNR ~31 dB — jauh lebih baik.
Hasil segmentasi kita akan dibandingkan dengan kedua angka ini.

---

## 1.6 Dataset

Dataset dari Kaggle: `mohamedhanyyy/chest-ctscan-images`

```python
import kagglehub
path = kagglehub.dataset_download("mohamedhanyyy/chest-ctscan-images")
```

Berisi gambar CT-Scan paru-paru dengan 4 kategori:
- **normal** → paru-paru sehat
- **adenocarcinoma** → kanker jenis adenokarsinoma
- **large.cell.carcinoma** → kanker sel besar
- **squamous.cell.carcinoma** → kanker sel skuamosa

Untuk proyek ini, kita fokus memproses gambarnya saja (bukan klasifikasi),
jadi semua kelas diperlakukan sama — tujuannya segmentasi area kanker.

---

## 1.7 Ringkasan ATM

| | Jurnal 1 | Jurnal 2 | Penelitian Kita |
|--|---------|---------|----------------|
| **AMATI** | Edge detection | Filter + Edge detection | ← dipelajari keduanya |
| **TIRU** | Studi kasus CT-Scan | Metrik MSE & PSNR | ← diadopsi |
| **MODIFIKASI** | - | - | Ganti edge detection → Segmentasi (CLAHE + Otsu + Morfologi) |

---

# BAGIAN 2 — ENVIRONMENT

Gunakan **Google Colab** (bukan lokal). Alasan utama:
- GPU gratis (meskipun tidak dibutuhkan untuk metode ini, Colab tetap lebih cepat)
- Tidak perlu install apapun, semua library sudah tersedia
- Hasil bisa langsung disimpan ke Google Drive

### Setup Awal di Colab

```python
# Cell 1 — Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import os
os.makedirs('/content/drive/MyDrive/ATM_KankerParu/results', exist_ok=True)
```

```python
# Cell 2 — Install kagglehub
!pip install kagglehub -q
```

```python
# Cell 3 — Import semua library
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

print("OpenCV version:", cv2.__version__)
```

---

# BAGIAN 3 — LANGKAH PENGERJAAN

---

## LANGKAH 1 — Download dan Eksplorasi Dataset

```python
import kagglehub

path = kagglehub.dataset_download("mohamedhanyyy/chest-ctscan-images")
print("Dataset ada di:", path)

# Eksplorasi struktur
TRAIN_DIR = os.path.join(path, 'train')
TEST_DIR  = os.path.join(path, 'test')

print("\n=== Jumlah Gambar per Kelas ===")
for folder in sorted(os.listdir(TEST_DIR)):
    folder_path = os.path.join(TEST_DIR, folder)
    if os.path.isdir(folder_path):
        jumlah = len(os.listdir(folder_path))
        print(f"  {folder}: {jumlah} gambar")
```

---

## LANGKAH 2 — Fungsi Preprocessing (CLAHE)

```python
def preprocessing(img_path, img_size=256):
    """
    Baca gambar, ubah ke grayscale, resize, lalu terapkan CLAHE.
    CLAHE menggantikan perbaikan kontras manual pada jurnal referensi.
    """
    # Baca gambar
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Gambar tidak ditemukan: {img_path}")

    # Ubah ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize ke ukuran seragam
    gray = cv2.resize(gray, (img_size, img_size))

    # Terapkan CLAHE (perbaikan kontras adaptif)
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hasil  = clahe.apply(gray)

    return gray, hasil  # kembalikan sebelum dan sesudah CLAHE
```

---

## LANGKAH 3 — Fungsi Segmentasi (Otsu + Morfologi)

```python
def segmentasi(img_clahe):
    """
    Segmentasi gambar menggunakan Otsu Thresholding + Operasi Morfologi.
    Ini adalah MODIFIKASI utama dari edge detection di jurnal referensi.
    """
    # Otsu Thresholding — threshold otomatis
    _, mask_otsu = cv2.threshold(
        img_clahe, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Operasi Morfologi — bersihkan hasil
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Opening: buang noise kecil di luar area kanker
    opening = cv2.morphologyEx(mask_otsu, cv2.MORPH_OPEN, kernel)

    # Closing: tutup lubang kecil di dalam area kanker
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    return mask_otsu, opening, closing
```

---

## LANGKAH 4 — Fungsi Hitung MSE dan PSNR

```python
def hitung_mse(img_asli, img_hasil):
    """MSE antara gambar asli dan hasil segmentasi."""
    asli  = img_asli.astype(np.float64)
    hasil = img_hasil.astype(np.float64)
    return np.mean((asli - hasil) ** 2)

def hitung_psnr(mse, max_pixel=255.0):
    """PSNR dari nilai MSE."""
    if mse == 0:
        return float('inf')
    return 10 * np.log10((max_pixel ** 2) / mse)
```

---

## LANGKAH 5 — Proses Semua Gambar dan Tampilkan Hasil

```python
def proses_satu_gambar(img_path, simpan_path=None):
    """
    Pipeline lengkap: baca → preprocessing → segmentasi → hitung metrik → tampilkan.
    """
    # === Preprocessing ===
    gray, clahe_img = preprocessing(img_path)

    # === Segmentasi ===
    mask_otsu, mask_opening, mask_closing = segmentasi(clahe_img)

    # === Hitung Metrik ===
    mse_otsu    = hitung_mse(gray, mask_otsu)
    psnr_otsu   = hitung_psnr(mse_otsu)
    mse_closing = hitung_mse(gray, mask_closing)
    psnr_closing= hitung_psnr(mse_closing)

    # === Visualisasi ===
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    judul = ['Asli (Grayscale)', 'Setelah CLAHE', 'Otsu Threshold',
             'Setelah Opening', 'Setelah Closing (Final)']
    gambar= [gray, clahe_img, mask_otsu, mask_opening, mask_closing]

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
        'file'        : nama_file,
        'mse_otsu'    : round(mse_otsu,    2),
        'psnr_otsu'   : round(psnr_otsu,   2),
        'mse_final'   : round(mse_closing, 2),
        'psnr_final'  : round(psnr_closing,2)
    }
```

---

## LANGKAH 6 — Jalankan untuk Semua Kelas

```python
KELAS = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']
HASIL_SEMUA = []

for kelas in KELAS:
    folder = os.path.join(TEST_DIR, kelas)
    if not os.path.isdir(folder):
        continue

    # Ambil 3 gambar pertama dari tiap kelas sebagai sampel
    gambar_list = sorted(os.listdir(folder))[:3]

    print(f"\n=== Kelas: {kelas.upper()} ===")

    for nama_gambar in gambar_list:
        img_path   = os.path.join(folder, nama_gambar)
        simpan     = f'/content/drive/MyDrive/ATM_KankerParu/results/{kelas}_{nama_gambar}.png'

        hasil = proses_satu_gambar(img_path, simpan_path=simpan)
        hasil['kelas'] = kelas
        HASIL_SEMUA.append(hasil)

        print(f"  {nama_gambar} → MSE: {hasil['mse_final']}, PSNR: {hasil['psnr_final']} dB")
```

---

## LANGKAH 7 — Buat Tabel Ringkasan Metrik

```python
import pandas as pd

df = pd.DataFrame(HASIL_SEMUA)

print("\n=== TABEL HASIL SEGMENTASI ===")
print(df[['kelas', 'file', 'mse_otsu', 'psnr_otsu', 'mse_final', 'psnr_final']].to_string(index=False))

# Rata-rata per kelas
print("\n=== RATA-RATA PER KELAS ===")
print(df.groupby('kelas')[['mse_final', 'psnr_final']].mean().round(2).to_string())

# Rata-rata keseluruhan
print(f"\n=== RATA-RATA KESELURUHAN ===")
print(f"MSE  rata-rata: {df['mse_final'].mean():.2f}")
print(f"PSNR rata-rata: {df['psnr_final'].mean():.2f} dB")

# Simpan ke CSV
df.to_csv('/content/drive/MyDrive/ATM_KankerParu/results/metrics_table.csv', index=False)
print("\nTabel tersimpan ke Google Drive.")
```

---

## LANGKAH 8 — Tabel Perbandingan Final dengan Jurnal

```python
print("""
=== TABEL PERBANDINGAN ATM ===

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
| Tools              | MATLAB R2017b           | MATLAB R2015a           | Python + OpenCV           |
+--------------------+-------------------------+-------------------------+---------------------------+
| MSE terbaik        | 37.278 (Canny)          | 47 (Median filter)      | [isi setelah eksperimen]  |
+--------------------+-------------------------+-------------------------+---------------------------+
| PSNR terbaik       | 2,43 dB                 | 31 dB                   | [isi setelah eksperimen]  |
+--------------------+-------------------------+-------------------------+---------------------------+
""")
```

---

# BAGIAN 4 — ALUR KERJA KESELURUHAN

```
Gambar CT-Scan (input)
        ↓
[Grayscale + Resize]          ← standar preprocessing CV
        ↓
[CLAHE]                       ← TIRU: perbaikan kontras (modifikasi dari jurnal)
        ↓
[Otsu Thresholding]           ← MODIFIKASI: ganti edge detection → segmentasi
        ↓
[Opening → Closing]           ← MODIFIKASI: bersihkan hasil dengan morfologi
        ↓
[Hitung MSE & PSNR]           ← TIRU: metrik sama dengan jurnal referensi
        ↓
[Bandingkan dengan jurnal]    ← output akhir untuk laporan
```

---

# BAGIAN 5 — STRUKTUR OUTPUT

```
ATM_KankerParu/                         (di Google Drive)
└── results/
    ├── adenocarcinoma_[nama].png        ← visualisasi per gambar
    ├── normal_[nama].png
    ├── large.cell.carcinoma_[nama].png
    ├── squamous.cell.carcinoma_[nama].png
    └── metrics_table.csv               ← semua nilai MSE & PSNR
```

---

# BAGIAN 6 — CHECKLIST SEBELUM SELESAI

- [ ] Dataset berhasil didownload dan jumlah gambar per kelas tercatat
- [ ] CLAHE berjalan dan perbedaan sebelum/sesudah terlihat di visualisasi
- [ ] Otsu Thresholding menghasilkan mask yang masuk akal
- [ ] Morfologi (opening + closing) memperbaiki kualitas mask
- [ ] MSE dan PSNR berhasil dihitung untuk setiap gambar
- [ ] Tabel ringkasan metrik tersimpan sebagai CSV
- [ ] Tabel perbandingan ATM sudah diisi lengkap (kolom "Penelitian Ini")
- [ ] Minimal 3 visualisasi per kelas tersimpan di Drive

---

# BAGIAN 7 — REFERENSI

| Sumber | Link / Keterangan |
|--------|------------------|
| Dataset Kaggle | https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images |
| OpenCV Thresholding | https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html |
| OpenCV Morfologi | https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html |
| OpenCV CLAHE | https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html |
| Jurnal 1 | Husni & Adrial, Jurnal Fisika Unand Vol.12 No.1 (2023) |
| Jurnal 2 | Fendriani dkk, JoP Vol.8 No.2 (2023) |

---

*Panduan ini untuk agent Antigravity — ATM Computer Vision, murni OpenCV tanpa Deep Learning.*
