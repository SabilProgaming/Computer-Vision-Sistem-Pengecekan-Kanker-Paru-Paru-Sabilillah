# 📖 Dokumentasi Proyek ATM — Segmentasi Kanker Paru-Paru

## Mata Kuliah: Computer Vision
## Metode: CLAHE + Otsu Thresholding + Morfologi (OpenCV)

---

## 1. Pendahuluan

Proyek ini menggunakan pendekatan **ATM (Amati, Tiru, Modifikasi)** terhadap dua jurnal referensi tentang pengolahan citra CT-Scan kanker paru-paru.

### Apa yang Diamati dari Jurnal?

| | Jurnal 1 (Husni & Adrial, 2023) | Jurnal 2 (Fendriani dkk, 2023) |
|--|--------------------------------|-------------------------------|
| **Metode** | Edge Detection (Robert, Sobel, Prewitt, Canny) | Canny + Filter (Mean, Median, Gaussian) |
| **Data** | 5 pasien CT Simulator, RS Unand | 6 pasien CT-Scan, RS Raden Mattaher Jambi |
| **Tools** | MATLAB R2017b | MATLAB R2015a |
| **Hasil terbaik** | Canny → MSE 37.278, PSNR 2,43 dB | Median Filter → MSE 47, PSNR 31 dB |
| **Kelemahan** | Hanya menghasilkan garis tepi, tidak bisa memisahkan area kanker | Masih berupa garis tepi, bukan area solid |

### Apa yang Ditiru?
- **Objek penelitian:** Citra CT-Scan kanker paru-paru
- **Metrik evaluasi:** MSE dan PSNR (rumus dan interpretasi yang sama)
- **Konsep preprocessing:** Perbaikan kontras sebelum proses utama

### Apa yang Dimodifikasi?
- **Edge Detection → Segmentasi:** Garis tepi diganti dengan area terisi penuh
- **Kontras manual → CLAHE:** Perbaikan kontras adaptif otomatis
- **MATLAB → Python + OpenCV:** Tools modern dan gratis
- **Penambahan Morfologi:** Operasi pembersihan hasil yang tidak ada di jurnal

---

## 2. Alur Kerja Proyek (Pipeline)

```
Gambar CT-Scan (input)
        ↓
[1. Grayscale + Resize]          ← Standar preprocessing
        ↓
[2. CLAHE]                       ← TIRU: perbaikan kontras (modifikasi dari jurnal)
        ↓
[3. Otsu Thresholding]           ← MODIFIKASI: ganti edge detection → segmentasi
        ↓
[4. Opening → Closing]           ← MODIFIKASI: bersihkan hasil dengan morfologi
        ↓
[5. Hitung MSE & PSNR]           ← TIRU: metrik sama dengan jurnal referensi
        ↓
[6. Visualisasi & Perbandingan]  ← Output akhir untuk laporan
```

---

## 3. Penjelasan Setiap Blok Kode

### CELL 1 — Mount Google Drive
**Tujuan:** Menghubungkan Google Colab ke Google Drive.

**Kenapa diperlukan?** Colab adalah lingkungan sementara — semua file akan hilang saat session berakhir. Dengan mount Drive, hasil visualisasi dan tabel metrik tersimpan permanen.

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### CELL 2 — Install Library Tambahan
**Tujuan:** Menginstall `kagglehub` untuk mendownload dataset dari Kaggle.

**Catatan:** Library utama (OpenCV, NumPy, Matplotlib) sudah terinstall secara default di Google Colab.

---

### CELL 3 — Import Library
**Tujuan:** Memuat semua library yang dibutuhkan ke dalam memori.

| Library | Fungsi dalam Proyek |
|---------|-------------------|
| `cv2` (OpenCV) | Pemrosesan gambar: CLAHE, thresholding, morfologi |
| `numpy` | Operasi matematika pada array pixel (MSE, PSNR) |
| `matplotlib` | Menampilkan dan menyimpan visualisasi gambar |
| `os`, `glob` | Navigasi file dan folder dataset |
| `pandas` | Membuat tabel ringkasan metrik dalam format terstruktur |

---

### CELL 4 — Download & Eksplorasi Dataset
**Tujuan:** Mengunduh dataset CT-Scan dari Kaggle dan menghitung distribusi gambar.

**Dataset:** `mohamedhanyyy/chest-ctscan-images`

**Struktur dataset:**
```
chest-ctscan-images/
├── train/
│   ├── adenocarcinoma/           ← Kanker jenis Adenocarcinoma
│   ├── large.cell.carcinoma/     ← Kanker sel besar
│   ├── normal/                   ← Paru-paru sehat
│   └── squamous.cell.carcinoma/  ← Kanker sel skuamosa
└── test/
    ├── (sama seperti di atas)
```

**Kenapa pakai folder `test`?** Proyek ini bukan klasifikasi — kita cukup mengambil sampel gambar untuk diproses dengan pipeline segmentasi.

---

### CELL 5 — Fungsi `preprocessing()`
**Tujuan:** Menyiapkan gambar CT-Scan sebelum proses segmentasi.

| No | Operasi | Penjelasan |
|----|---------|-----------|
| 1 | `cv2.imread()` | Membaca file gambar dari disk |
| 2 | `cv2.cvtColor(BGR2GRAY)` | Mengubah gambar berwarna ke grayscale (1 channel) |
| 3 | `cv2.resize(256, 256)` | Menyeragamkan ukuran semua gambar |
| 4 | `cv2.createCLAHE().apply()` | Meningkatkan kontras secara adaptif |

**Tentang CLAHE:**
- **Apa:** Contrast Limited Adaptive Histogram Equalization
- **Cara kerja:** Membagi gambar ke grid 8×8, lalu meratakan histogram di setiap grid secara terpisah
- **clipLimit=2.0:** Membatasi amplifikasi kontras agar tidak berlebihan
- **Keunggulan vs jurnal:** Jurnal menggunakan mapping manual (0.2–0.45 → 0–255), CLAHE bekerja otomatis dan adaptif

---

### CELL 6 — Fungsi `segmentasi()`
**Tujuan:** Memisahkan area kanker dari latar belakang — **MODIFIKASI UTAMA** dari jurnal.

#### A. Otsu Thresholding
```python
_, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```
- Menganalisis histogram dan secara otomatis menentukan nilai threshold (T) optimal
- Pixel > T → putih (255, area terang/kanker), Pixel ≤ T → hitam (0, latar belakang)

> **Beda dengan Edge Detection (jurnal):**
> Edge Detection → **garis tepi** (outline). Segmentasi → **area terisi** (filled region).

#### B. Opening (Erosi → Dilasi)
- Menghilangkan titik putih kecil (noise) di luar area kanker

#### C. Closing (Dilasi → Erosi)
- Menutup lubang hitam kecil di dalam area kanker

**Kernel elips 5×5:** Dipilih karena bentuk tumor paru cenderung membulat.

---

### CELL 7 — Fungsi `hitung_mse()` dan `hitung_psnr()`
**Tujuan:** Menghitung metrik evaluasi yang sama dengan jurnal referensi.

| Metrik | Rumus | Interpretasi |
|--------|-------|-------------|
| MSE | `(1/MN) × Σ[I(i,j) - K(i,j)]²` | Rendah = mirip asli; Tinggi = sangat berbeda |
| PSNR | `10 × log₁₀(255² / MSE)` | > 30 dB = baik; < 10 dB = buruk |

**Pembanding dari jurnal:**
- Jurnal 1 (Canny): MSE=37.278, PSNR=2,43 dB
- Jurnal 2 (Median Filter): MSE=47, PSNR=31 dB

---

### CELL 8 — Fungsi `proses_satu_gambar()`
**Tujuan:** Pipeline lengkap untuk satu gambar, menghasilkan visualisasi 5 panel:

| Panel | Isi |
|-------|-----|
| 1 | Asli (Grayscale) |
| 2 | Setelah CLAHE |
| 3 | Otsu Threshold |
| 4 | Setelah Opening |
| 5 | Setelah Closing (Final) |

---

### CELL 9 — Loop Semua Kelas
**Tujuan:** Memproses 3 sampel × 4 kelas = 12 gambar total.

### CELL 10 — Tabel Ringkasan Metrik
**Tujuan:** Menampilkan tabel metrik + rata-rata per kelas + simpan CSV.

### CELL 11 — Tabel Perbandingan ATM
**Tujuan:** Perbandingan akhir antara jurnal dan penelitian kita.

---

## 4. Cara Menjalankan Proyek

1. Buka [Google Colab](https://colab.research.google.com)
2. Buat notebook baru: **File → New Notebook**
3. Salin isi `main.py` — pisahkan setiap `CELL` menjadi cell terpisah di Colab
4. **Cell 2** (`!pip install kagglehub -q`) harus dijalankan sebagai cell terpisah
5. Jalankan berurutan dari atas ke bawah
6. Cek hasil di Google Drive → `ATM_KankerParu/results/`

---

## 5. Catatan Perubahan (Changelog)

| Versi | Tanggal | Perubahan |
|-------|---------|-----------|
| v1.0 | 2026-05-03 | Versi awal (salah: menggunakan CNN/VGG16/Grad-CAM) |
| v2.0 | 2026-05-03 | **Ditulis ulang** sesuai panduan yang benar: CLAHE + Otsu + Morfologi, murni CV klasik |

### Detail Perubahan v1.0 → v2.0:
- ❌ Dihapus: CNN, VGG16, Transfer Learning, Grad-CAM, TensorFlow
- ✅ Ditambah: Otsu Thresholding, Operasi Morfologi
- ✅ Dipertahankan: CLAHE, MSE, PSNR
- 🔄 Diubah: Platform dari lokal → Google Colab
- 🔄 Diubah: Tujuan dari klasifikasi → segmentasi area kanker

---

## 6. Referensi

| Sumber | Link / Keterangan |
|--------|-------------------|
| Dataset Kaggle | https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images |
| OpenCV Thresholding | https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html |
| OpenCV Morfologi | https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html |
| OpenCV CLAHE | https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html |
| Jurnal 1 | Husni & Adrial, Jurnal Fisika Unand Vol.12 No.1 (2023) |
| Jurnal 2 | Fendriani dkk, JoP Vol.8 No.2 (2023) |
