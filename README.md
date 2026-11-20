# Auto YouTube Clipper (Windows Edition)

Sistem otomatisasi Python untuk memproses video YouTube panjang (misal: 1 jam gameplay/stream) menjadi maksimal **10 video vertikal (9:16 / 1080x1920)** siap unggah untuk **YouTube Shorts** dan **TikTok**.

---

## 🚀 Fitur Utama
1. **Ekstraksi Otomatis:** Mengambil video 1080p dan metadata nama channel secara otomatis menggunakan `yt-dlp`.
2. **Deteksi Facecam Streamer (OpenCV):** Mendeteksi posisi webcam streamer di pojok layar menggunakan Haar Cascade dan mengabaikan karakter game di tengah.
3. **Transkripsi Lokal (faster-whisper):** Speech-to-Text lokal berkecepatan tinggi dengan timestamp per kata (`word_timestamps=True`).
4. **Kurasi Momen Cerdas (Gemini 3.8 Flash):** Menganalisis transkrip untuk memilih momen paling lucu, panik, atau tegang (durasi 45–60 detik) serta menghasilkan judul SEO dan headline hook viral.
5. **Layout Dual Mode (FFmpeg):**
   - **Mode Streamer:** Zona Atas Facecam (`1080x640`) + Zona Tengah Gameplay (`1080x960`) + Zona Bawah Headline Hook & Safe Zone TikTok/Shorts (`1080x320`).
   - **Mode Biasa:** *Center crop* vertikal 9:16 standar layar penuh.
6. **Subtitle Dinamis (Hardsub ASS):** Subtitle ucapan otomatis ditempelkan tepat di batas bawah facecam streamer (`y ≈ 585`) atau di area aman mode biasa dengan tampilan 3–4 kata per baris.

---

## 🛠️ Prasyarat Sistem
1. **Python 3.10+** terpasang di Windows.
2. **FFmpeg CLI:**
   ### Cara Mendapatkan ffmpeg.exe yang Benar:
   **Opsi 1: Unduh Manual (Paling Mudah & Cepat)**
   - Klik tautan unduh langsung resmi untuk Windows ini: 👉 [Download FFmpeg Essentials (ZIP)](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip)
   - Buka dan Ekstrak (*Extract*) file zip tersebut.
   - Masuk ke dalam folder `bin` hasil ekstrak. Di sana Anda akan melihat 3 file:
     - `ffmpeg.exe`
     - `ffplay.exe`
     - `ffprobe.exe`
   - Salin (*Copy*) ketiga file tersebut dan Tempel (*Paste*) langsung ke folder proyek Anda di: 📁 `D:\MY PROJECT\auto_youtube_clipper\`
   *(Skrip `auto_clipper.py` sudah diprogram untuk otomatis mendeteksi jika `ffmpeg.exe` ditaruh langsung di folder tersebut).*

   **Opsi 2: Menggunakan Terminal Windows (Winget)**
   ```powershell
   winget install Gyan.FFmpeg
   ```
3. *(Opsional)* **NVIDIA GPU + CUDA Toolkit** jika ingin akselerasi hardware saat transkripsi Whisper.

---

## 📦 Panduan Instalasi

Buka terminal (CMD / PowerShell) di folder ini, lalu jalankan:

```powershell
# 1. Pasang paket dependensi
pip install -r requirements.txt

# 2. Siapkan file konfigurasi .env
copy .env.example .env
```

Buka file `.env` dan masukkan API Key Google Gemini Anda:
```env
GEMINI_API_KEY=AIzaSy...
```
*(Dapatkan API Key gratis di [Google AI Studio](https://aistudio.google.com/))*

---

## 🎮 Cara Menjalankan

### Cara 1: Menggunakan Web Studio UI (Tampilan Browser Interaktif)
Ini adalah cara yang **paling direkomendasikan** dengan antarmuka visual modern (Tailwind CSS + Vue 3):
1. Cukup **klik ganda (double-click)** pada file **`start_web.bat`**, atau jalankan di terminal:
   ```powershell
   python app.py
   ```
2. Browser Anda akan terbuka otomatis di alamat: **`http://localhost:8000`**.
3. Fitur di Web UI:
   * **Animasi Progress Bar Real-Time (0–100%)** untuk setiap tahapan proses (Download, STT CUDA, AI Gemini 3.8, dan Render FFmpeg).
   * **Mode Review & Kustomisasi:** Anda dapat mengedit teks *Headline Hook*, mengatur durasi, dan memilih klip mana saja yang mau dirender.
   * **Player Video 9:16:** Bisa langsung memutar video klip yang selesai dan mengunduhnya dengan tombol satu klik.

---

### Cara 2: Menggunakan Terminal / CLI Klasik
Cukup **klik ganda** file `run.bat` atau jalankan via terminal:
```powershell
# Mode 1: Streamer (Facecam di atas, Gameplay di tengah)
python auto_clipper.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode streamer

# Mode 2: Biasa (Center Crop 9:16 Full Screen)
python auto_clipper.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode biasa
```

---

## 📂 Struktur Output

Setelah proses selesai, semua klip dan informasinya akan tersimpan di dalam folder `output_clips/`:

```
output_clips/
├── clip_01.mp4           <- Klip 9:16 siap unggah (1080x1920)
├── clip_02.mp4
├── ...
├── clip_10.mp4
└── info_klip.txt         <- Rangkuman judul SEO, durasi, hook, dan alasan momen dipilih
```

Contoh isi `info_klip.txt`:
```text
INFORMASI HASIL CLIPPING YOUTUBE
Channel Sumber : fadilahGS
Mode Layout    : Streamer
Total Klip     : 10 klip
============================================================
Klip 01: clip_01.mp4
  Judul SEO : Detik-detik Panik Dikejar Polisi di Los Santos!
  Durasi    : 145.2s s/d 198.5s (53.3s)
  Hook Teks : "Momen Santai Sebelum Kekacauan Dimulai!"
  Alasan    : Ada kejutan mendadak dan reaksi lucu streamer saat dikepung.
------------------------------------------------------------
```

---

## 📐 Tata Letak Visual (Mode Streamer)
Kanvas: `1080 x 1920` (Rasio 9:16)
- **Zona 1 (y: 0 - 640):** Facecam Streamer (`1080x640`), Credit `CC: YouTube / {Channel}` di pojok kiri atas, dan Subtitle Ucapan di `y ≈ 585`.
- **Zona 2 (y: 640 - 1600):** Layar gameplay utama (`1080x960`).
- **Zona 3 (y: 1600 - 1920):** Headline Hook viral di `y = 1635`, menyisakan Safe Zone kosong di `y = 1770 - 1920` agar tidak tertutup antarmuka TikTok / Shorts.
