"""
Auto YouTube Clipper (Windows Edition)
======================================
Mengubah video YouTube panjang menjadi klip vertikal siap unggah (Shorts / TikTok)
berdasarkan spesifikasi: spesifikasi_prompt_auto_youtube_clipper.md
"""

import os
import sys
import shutil
import subprocess
import json
import re
import time
import math
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables dari file .env jika ada
load_dotenv()

def setup_cuda_dlls():
    """Menambahkan direktori library NVIDIA cuBLAS dan cuDNN pip ke PATH Windows."""
    import site
    site_dirs = []
    try:
        site_dirs.extend(site.getsitepackages())
    except Exception:
        pass
    if hasattr(site, "getusersitepackages"):
        try:
            site_dirs.append(site.getusersitepackages())
        except Exception:
            pass

    for s_dir in site_dirs:
        nvidia_path = Path(s_dir) / "nvidia"
        if nvidia_path.exists():
            for sub in ["cublas", "cudnn", "cuda_runtime"]:
                bin_path = nvidia_path / sub / "bin"
                if bin_path.exists():
                    str_bin = str(bin_path.resolve())
                    if str_bin not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = str_bin + os.pathsep + os.environ.get("PATH", "")
                    if hasattr(os, "add_dll_directory"):
                        try:
                            os.add_dll_directory(str_bin)
                        except Exception:
                            pass

setup_cuda_dlls()

# ==============================================================================
# KONFIGURASI PENGGUNA
# ==============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Model Whisper: 'tiny', 'base', 'small', 'medium', 'large-v3'
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

# Device akselerasi: 'cuda' (NVIDIA GPU) atau 'cpu'
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = "float16" if WHISPER_DEVICE.lower() == "cuda" else "int8"

# Pengaturan Klip
MAX_CLIPS = 10
MIN_CLIP_DURATION = 45  # Detik
MAX_CLIP_DURATION = 60  # Detik

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output_clips"
TEMP_DIR = BASE_DIR / "temp_work"
FONTS_DIR = BASE_DIR / "fonts"

# Daftar Preset Font Rekomendasi untuk Judul Hook (#FFD700)
FONT_PRESETS: Dict[str, Dict[str, Any]] = {
    "anton": {
        "label": "Anton (Tebal & Gagah - Gaming/Stream)",
        "file": "Anton-Regular.ttf",
        "base_size": 52,
        "max_chars_normal": 35,
        "uppercase": False,
    },
    "bebas": {
        "label": "Bebas Neue (Kapital Ramping & Tegas)",
        "file": "BebasNeue-Regular.ttf",
        "base_size": 56,
        "max_chars_normal": 36,
        "uppercase": True,
    },
    "impact": {
        "label": "Impact (Klasik Thumbnail & Meme)",
        "file": "impact.ttf",
        "base_size": 50,
        "max_chars_normal": 35,
        "uppercase": False,
    },
    "montserrat": {
        "label": "Montserrat Black (Bersih & Viral Shorts)",
        "file": "Montserrat-Black.ttf",
        "base_size": 40,
        "max_chars_normal": 29,
        "uppercase": False,
    },
    "poppins": {
        "label": "Poppins Bold (Geometris & Modern)",
        "file": "Poppins-Bold.ttf",
        "base_size": 40,
        "max_chars_normal": 30,
        "uppercase": False,
    },
    "opensans": {
        "label": "Open Sans ExtraBold (Jelas & Edukasi)",
        "file": "OpenSans-ExtraBold.ttf",
        "base_size": 40,
        "max_chars_normal": 30,
        "uppercase": False,
    },
}


# ==============================================================================
# PYDANTIC SCHEMA UNTUK OUTPUT TERSTRUKTUR GEMINI
# ==============================================================================
class ClipSegment(BaseModel):
    start_seconds: float = Field(
        description="Timestamp awal klip dalam detik (misal: 120.5)"
    )
    end_seconds: float = Field(
        description="Timestamp akhir klip dalam detik (durasi wajib 45 - 60 detik)"
    )
    title: str = Field(
        description="Judul SEO menarik dan viral untuk YouTube Shorts / TikTok"
    )
    hook_headline: str = Field(
        description="Kalimat pancingan singkat maksimal 2 baris untuk ditampilkan di video"
    )
    reason: str = Field(
        description="Alasan momen ini dipilih (lucu, panik, tegang, emosional, dll)"
    )


class ClipAnalysisResult(BaseModel):
    clips: List[ClipSegment] = Field(
        description="Daftar maksimal 10 segmen klip terbaik dengan durasi 45-60 detik"
    )


# ==============================================================================
# UTILITY & DEPENDENCY CHECKS
# ==============================================================================
def setup_yt_dlp():
    """Menyiapkan yt_dlp dan membungkam peringatan Deprecation Python 3.10."""
    try:
        import yt_dlp
        import yt_dlp.update
        yt_dlp.update._get_system_deprecation = lambda: None
        if 'yt_dlp.YoutubeDL' in sys.modules:
            sys.modules['yt_dlp.YoutubeDL']._get_system_deprecation = lambda: None
    except Exception:
        pass


def check_dependencies():
    """Memeriksa keberadaan ffmpeg dan dependensi sistem."""
    print("Memeriksa dependensi sistem...")
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        # Cek lokasi umum ffmpeg di Windows
        common_paths = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"D:\ffmpeg\bin\ffmpeg.exe",
            r"D:\MY_PROJECT\bin\ffmpeg.exe",
            str(BASE_DIR / "ffmpeg.exe")
        ]
        found = False
        for p in common_paths:
            if os.path.exists(p):
                os.environ["PATH"] += os.pathsep + os.path.dirname(p)
                found = True
                print(f"[OK] FFmpeg ditemukan di: {p}")
                break
        
        if not found:
            print("\n[PERINGATAN] 'ffmpeg' tidak ditemukan di PATH sistem!")
            print("Silakan unduh FFmpeg untuk Windows dari https://ffmpeg.org/download.html")
            print("Atau pasang via terminal: winget install Gyan.FFmpeg")
            print("Atau salin ffmpeg.exe langsung ke folder project ini.\n")
            sys.exit(1)
    else:
        print(f"[OK] FFmpeg terdeteksi: {ffmpeg_path}")

    try:
        setup_yt_dlp()
        import yt_dlp
        print("[OK] yt-dlp terpasang.")
    except ImportError:
        print("[ERROR] yt-dlp belum terpasang. Jalankan: pip install yt-dlp")
        sys.exit(1)

    try:
        from faster_whisper import WhisperModel
        print("[OK] faster-whisper terpasang.")
    except ImportError:
        print("[ERROR] faster-whisper belum terpasang. Jalankan: pip install faster-whisper")
        sys.exit(1)

    try:
        import cv2
        print("[OK] opencv-python terpasang.")
    except ImportError:
        print("[ERROR] opencv-python belum terpasang. Jalankan: pip install opencv-python")
        sys.exit(1)


def escape_ffmpeg_text(text: str) -> str:
    """Escape karakter khusus untuk filter drawtext FFmpeg di Windows."""
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "\u2019")  # ganti single quote dengan apostrophe aman
    text = text.replace(":", "\\:")
    text = text.replace("%", "\\%")
    text = text.replace("\n", "\\\n")
    return text


def format_seconds(seconds: float) -> str:
    """Format detik menjadi format HH:MM:SS.mmm"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def format_ass_time(seconds: float) -> str:
    """Format detik menjadi format ASS H:MM:SS.cc"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    centis = int(round((s - int(s)) * 100))
    if centis >= 100:
        centis = 99
    return f"{int(h):01d}:{int(m):02d}:{int(s):02d}.{centis:02d}"


# ==============================================================================
# STEP 2: METADATA & DOWNLOAD YOUTUBE
# ==============================================================================
def get_channel_name(youtube_url: str) -> str:
    """Mengekstrak nama channel asli via yt_dlp."""
    print("\nMengekstrak metadata channel...")
    setup_yt_dlp()
    try:
        import yt_dlp
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'ffmpeg_location': str(BASE_DIR),
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            channel = info.get('channel') or info.get('uploader') or info.get('channel_id')
            if channel:
                print(f"[OK] Channel Sumber: {channel}")
                return str(channel).strip()
    except Exception as e:
        print(f"[WARN] Gagal mendapatkan nama channel secara otomatis: {e}")

    return "Creator"


def download_youtube_video(youtube_url: str, output_path: Path, progress_callback=None) -> Path:
    """Mengunduh video YouTube dalam format MP4 1080p ke file sementara menggunakan library yt_dlp."""
    print(f"\nMengunduh video sumber dari YouTube ke {output_path.name}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        try:
            output_path.unlink()
        except Exception:
            pass

    setup_yt_dlp()
    import yt_dlp

    def ytdl_progress_hook(d):
        if progress_callback:
            status = d.get('status')
            if status == 'downloading':
                downloaded = d.get('downloaded_bytes', 0)
                total = d.get('total_bytes') or d.get('total_bytes_estimate') or 1
                percent = min(100.0, round((downloaded / total) * 100, 1))
                speed = d.get('speed') or 0
                speed_str = f"{speed / (1024*1024):.1f} MB/s" if speed > 0 else "0 MB/s"
                downloaded_mb = f"{downloaded / (1024*1024):.1f} MB"
                total_mb = f"{total / (1024*1024):.1f} MB" if total > 1 else ""
                eta = d.get('eta')
                eta_str = f"ETA {int(eta)}s" if eta else ""
                filename = Path(d.get('filename', '')).name
                progress_callback({
                    "status": "downloading",
                    "percent": percent,
                    "downloaded_mb": downloaded_mb,
                    "total_mb": total_mb,
                    "speed": speed_str,
                    "eta": eta_str,
                    "filename": filename
                })
            elif status == 'finished':
                filename = Path(d.get('filename', '')).name
                progress_callback({
                    "status": "finished",
                    "percent": 100.0,
                    "downloaded_mb": "",
                    "total_mb": "",
                    "speed": "Selesai Unduh",
                    "eta": "",
                    "filename": filename,
                    "status_text": f"Format {filename} selesai diunduh. Menggabungkan / mengekstrak..."
                })

    def ytdl_postprocessor_hook(d):
        if progress_callback:
            status = d.get('status')
            if status == 'started':
                progress_callback({
                    "status": "extracting",
                    "percent": 100.0,
                    "downloaded_mb": "",
                    "total_mb": "",
                    "speed": "Mengekstrak / Merging",
                    "eta": "Proses FFmpeg muxing...",
                    "status_text": "Mengekstrak dan menggabungkan format video & audio..."
                })
            elif status == 'finished':
                progress_callback({
                    "status": "merged",
                    "percent": 100.0,
                    "downloaded_mb": "",
                    "total_mb": "",
                    "speed": "Selesai Ekstrak",
                    "eta": "",
                    "status_text": "Ekstraksi dan penggabungan video selesai 100%."
                })

    # Format: prioritas 1080p MP4, dengan fallback ke format terbaik yang kompatibel
    ydl_opts = {
        'format': 'bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/best[height<=1080]/best',
        'outtmpl': str(output_path),
        'merge_output_format': 'mp4',
        'noplaylist': True,
        'no_warnings': True,
        'ffmpeg_location': str(BASE_DIR),
        'quiet': False,
        'progress_hooks': [ytdl_progress_hook],
        'postprocessor_hooks': [ytdl_postprocessor_hook],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
    except Exception as err:
        print(f"[ERROR] Gagal mengunduh dengan format utama, mencoba mode kompatibilitas fallback...")
        fallback_opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': str(output_path),
            'merge_output_format': 'mp4',
            'noplaylist': True,
            'no_warnings': True,
            'ffmpeg_location': str(BASE_DIR),
            'quiet': False,
            'progress_hooks': [ytdl_progress_hook],
            'postprocessor_hooks': [ytdl_postprocessor_hook],
        }
        with yt_dlp.YoutubeDL(fallback_opts) as ydl:
            ydl.download([youtube_url])

    if not output_path.exists():
        raise FileNotFoundError(f"Gagal mengunduh video ke {output_path}")

    print(f"\n[OK] Video berhasil diunduh ({output_path.stat().st_size / (1024*1024):.1f} MB)")
    return output_path


def get_video_dimensions(video_path: Path) -> Tuple[int, int, float]:
    """Mendapatkan lebar, tinggi, dan durasi video menggunakan ffprobe / OpenCV."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Tidak dapat membuka video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps
    cap.release()
    return width, height, duration


# ==============================================================================
# STEP 3: DETEKSI WAJAH & WEBCAM STREAMER
# ==============================================================================
def detect_streamer_facecam(
    video_path: Path,
    video_w: int,
    video_h: int,
    facecam_pos: str = "auto"
) -> Tuple[int, int, int, int]:
    """
    Mendeteksi atau menentukan posisi facecam/webcam streamer.
    Target rasio crop facecam = 1080:640 = 1.6875.
    Mendukung deteksi otomatis (jika CascadeClassifier tersedia) atau posisi manual.
    """
    import cv2
    target_aspect = 1080.0 / 640.0  # 1.6875
    fallback_h = int(video_h * 0.40)
    fallback_w = int(fallback_h * target_aspect)

    def get_preset_box(pos: str) -> Tuple[int, int, int, int]:
        if pos == "bottom-left":
            return (0, video_h - fallback_h, fallback_w, fallback_h)
        elif pos == "top-right":
            return (video_w - fallback_w, 0, fallback_w, fallback_h)
        elif pos == "top-left":
            return (0, 0, fallback_w, fallback_h)
        else:  # default: bottom-right
            return (video_w - fallback_w, video_h - fallback_h, fallback_w, fallback_h)

    # Jika posisi eksplisit dipilih oleh pengguna
    if facecam_pos in ["bottom-right", "bottom-left", "top-right", "top-left"]:
        bx, by, bw, bh = get_preset_box(facecam_pos)
        print(f"[OK] Menggunakan posisi facecam preset '{facecam_pos}': x={bx}, y={by}, w={bw}, h={bh}")
        return (bx, by, bw, bh)

    print("\n[Mode Streamer] Memindai posisi webcam streamer dengan OpenCV...")
    detected_box = None

    try:
        # Periksa ketersediaan CascadeClassifier (OpenCV 4 vs OpenCV 5)
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                face_cascade = cv2.CascadeClassifier(cascade_path)
                cap = cv2.VideoCapture(str(video_path))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                sample_seconds = [30, 45, 60, 15, 90, 120]

                for sec in sample_seconds:
                    frame_idx = int(sec * fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        continue

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(int(video_h * 0.08), int(video_h * 0.08))
                    )

                    # Cari wajah di area pinggir layar (x < 35% atau x > 65%)
                    for (fx, fy, fw, fh) in faces:
                        center_x = fx + fw / 2.0
                        if center_x < (0.35 * video_w) or center_x > (0.65 * video_w):
                            print(f"[OK] Wajah streamer terdeteksi di detik ke-{sec}: x={fx}, y={fy}, w={fw}, h={fh}")
                            face_center_x = fx + (fw // 2)
                            face_center_y = fy + (fh // 2)

                            crop_h = int(fh * 2.8)
                            crop_w = int(crop_h * target_aspect)

                            if crop_h > video_h:
                                crop_h = video_h
                                crop_w = int(crop_h * target_aspect)

                            crop_x = int(face_center_x - (crop_w // 2))
                            crop_y = int(face_center_y - (crop_h * 0.35))

                            crop_x = max(0, min(crop_x, video_w - crop_w))
                            crop_y = max(0, min(crop_y, video_h - crop_h))

                            detected_box = (crop_x, crop_y, crop_w, crop_h)
                            break

                    if detected_box:
                        break

                cap.release()
        else:
            print(f"[INFO] Fitur CascadeClassifier tidak tersedia pada OpenCV v{cv2.__version__}.")
    except Exception as e:
        print(f"[WARN] Terjadi kendala saat pemindaian wajah: {e}")

    if detected_box:
        cx, cy, cw, ch = detected_box
        print(f"[OK] Area crop facecam terdeteksi: x={cx}, y={cy}, w={cw}, h={ch}")
        return detected_box

    # Fallback standar jika tidak terdeteksi otomatis: Pojok Kanan Bawah
    fb_x, fb_y, fb_w, fb_h = get_preset_box("bottom-right")
    print(f"[OK] Menggunakan area webcam default (Pojok Kanan Bawah): x={fb_x}, y={fb_y}, w={fb_w}, h={fb_h}")
    return (fb_x, fb_y, fb_w, fb_h)


# ==============================================================================
# STEP 4: TRANSKRIPSI FULL AUDIO LOKAL (FASTER-WHISPER)
# ==============================================================================
def transcribe_audio_with_whisper(
    video_path: Path,
    total_duration: float = 0.0,
    progress_callback=None
) -> List[Dict[str, Any]]:
    """
    Mentranskripsikan audio video secara lokal dengan faster-whisper.
    Mendukung deteksi GPU/CUDA otomatis dengan fallback mulus ke CPU jika library CUDA tidak ditemukan.
    """
    from faster_whisper import WhisperModel

    def try_transcribe(device_name: str, compute_t: str):
        print(f"\nMemulai transkripsi suara lokal dengan faster-whisper (model: {WHISPER_MODEL_SIZE}, device: {device_name})...")
        model = WhisperModel(WHISPER_MODEL_SIZE, device=device_name, compute_type=compute_t)
        segments_gen, info = model.transcribe(
            str(video_path),
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        print(f"[OK] Bahasa terdeteksi: {info.language} (probabilitas: {info.language_probability:.2f})")
        
        trans_data = []
        w_count = 0
        print("Mengekstrak kalimat dan timestamp per kata...")
        for seg in segments_gen:
            words_list = []
            if seg.words:
                for w in seg.words:
                    words_list.append({
                        "word": w.word.strip(),
                        "start": round(w.start, 2),
                        "end": round(w.end, 2)
                    })
                    w_count += 1
            trans_data.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
                "words": words_list
            })

            # Panggil progress callback jika ada
            if progress_callback and total_duration > 0:
                p_val = min(100.0, round((seg.end / total_duration) * 100, 1))
                progress_callback({
                    "percent": p_val,
                    "current_sec": round(seg.end, 1),
                    "total_sec": round(total_duration, 1),
                    "text": seg.text.strip(),
                    "words_count": w_count
                })

        return trans_data, w_count

    target_device = WHISPER_DEVICE.lower()
    target_compute = COMPUTE_TYPE

    try:
        data, total_words = try_transcribe(target_device, target_compute)
    except (RuntimeError, Exception) as err:
        err_msg = str(err)
        if target_device == "cuda" or "cublas" in err_msg.lower() or "cuda" in err_msg.lower():
            print(f"\n[WARN] Akselerasi CUDA gagal dijalankan ({err_msg}).")
            print("[INFO] CUDA Toolkit / cublas64_12.dll belum terpasang di sistem Windows.")
            print("[INFO] Beralih otomatis (fallback) ke mode CPU...")
            data, total_words = try_transcribe("cpu", "int8")
        else:
            raise err

    print(f"[OK] Transkripsi selesai: {len(data)} segmen, total {total_words} kata.")
    return data


# ==============================================================================
# STEP 5: PEMILIHAN MOMEN & PEMBUATAN HOOK (GEMINI AI)
# ==============================================================================
def analyze_moments_with_gemini(transcription_data: List[Dict[str, Any]], video_duration: float) -> List[ClipSegment]:
    """
    Mengirimkan transkrip ke Gemini 2.5 Flash untuk memilih maksimal 10 klip terbaik (45-60 detik).
    Menggunakan structured JSON schema via Pydantic.
    """
    api_key = GEMINI_API_KEY
    if not api_key:
        api_key = input("\n[INPUT] Masukkan GEMINI_API_KEY Anda: ").strip()

    if not api_key:
        raise ValueError("GEMINI_API_KEY wajib diisi untuk menganalisis momen terbaik!")

    print(f"\nMenganalisis momen terbaik dengan Gemini ({GEMINI_MODEL_NAME})...")

    # Format transkrip menjadi teks terkompresi dengan timestamp
    transcript_lines = []
    for item in transcription_data:
        start_fmt = format_seconds(item["start"])
        end_fmt = format_seconds(item["end"])
        transcript_lines.append(f"[{start_fmt} - {end_fmt}] ({item['start']:.1f}s) {item['text']}")

    full_transcript_text = "\n".join(transcript_lines)

    prompt = f"""
Anda adalah seorang ahli kurator konten dan editor video viral profesional untuk YouTube Shorts dan TikTok.
Tugas Anda adalah menganalisis transkrip audio dari video berdurasi total {video_duration:.1f} detik, lalu memilih maksimal {MAX_CLIPS} segmen momen terbaik.

Kriteria Pemilihan Momen:
1. Momen harus sangat menarik: lucu, panik, emosional, seru, tegang, adu argumen, atau ada plot twist.
2. Durasi tiap segmen WAJIB antara {MIN_CLIP_DURATION} detik hingga {MAX_CLIP_DURATION} detik (end_seconds - start_seconds >= {MIN_CLIP_DURATION} dan <= {MAX_CLIP_DURATION}).
3. Awal klip (start_seconds) harus dimulai dari kalimat atau konteks yang jelas, dan akhir klip (end_seconds) harus memiliki kesimpulan/klimaks yang memuaskan.
4. Buat 'hook_headline' yang sangat memancing rasa penasaran (maksimal 2 baris, singkat, padat, gaya bahasa santai/viral).
5. Buat 'title' yang SEO friendly untuk judul YouTube Shorts / TikTok.

Transkrip Video:
{full_transcript_text}
"""

    clips_result = []

    # Daftar kandidat model untuk dicoba berurutan jika model tertentu tidak tersedia
    candidate_models = []
    for m in [GEMINI_MODEL_NAME, "gemini-3.8-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
        if m and m not in candidate_models:
            candidate_models.append(m)

    success = False
    last_error = None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        for m_name in candidate_models:
            try:
                print(f"Menganalisis transkrip dengan model: {m_name}...")
                response = client.models.generate_content(
                    model=m_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ClipAnalysisResult,
                        temperature=0.7
                    )
                )

                parsed_data = json.loads(response.text)
                if isinstance(parsed_data, dict) and "clips" in parsed_data:
                    for c in parsed_data["clips"]:
                        clips_result.append(ClipSegment(**c))
                elif isinstance(parsed_data, list):
                    for c in parsed_data:
                        clips_result.append(ClipSegment(**c))

                if clips_result:
                    print(f"[OK] Sukses mendapatkan rekomendasi klip dari model: {m_name}")
                    success = True
                    break
            except Exception as err:
                last_error = err
                print(f"[INFO] Model {m_name} tidak dapat digunakan ({err}). Mencoba model alternatif...")
                continue

    except Exception as setup_err:
        raise RuntimeError(f"Gagal memuat pustaka google-genai: {setup_err}")

    if not success or not clips_result:
        raise RuntimeError(f"Gagal memanggil Gemini API: {last_error}")

    # Validasi dan penyesuaian durasi klip
    valid_clips = []
    for c in clips_result:
        # Batasi agar tidak melebihi durasi video
        start = max(0.0, c.start_seconds)
        end = min(video_duration, c.end_seconds)
        dur = end - start
        
        # Koreksi durasi jika sedikit di luar rentang
        if dur < MIN_CLIP_DURATION:
            end = min(video_duration, start + MIN_CLIP_DURATION)
        elif dur > MAX_CLIP_DURATION:
            end = start + MAX_CLIP_DURATION

        c.start_seconds = round(start, 2)
        c.end_seconds = round(end, 2)
        valid_clips.append(c)

    print(f"[OK] Gemini berhasil mengidentifikasi {len(valid_clips)} momen klip terbaik!")
    for idx, c in enumerate(valid_clips, 1):
        dur = c.end_seconds - c.start_seconds
        print(f"  {idx}. [{c.start_seconds}s - {c.end_seconds}s] ({dur:.1f}s) {c.title}")
        print(f"     Hook: \"{c.hook_headline}\"")

    return valid_clips[:MAX_CLIPS]


# ==============================================================================
# STEP 6: PEMBUATAN SUBTITLE (.ASS / .SRT) DENGAN KATA DINAMIS
# ==============================================================================
def generate_ass_subtitle_file(
    words_data: List[Dict[str, Any]],
    clip_start: float,
    clip_end: float,
    output_ass_path: Path,
    mode: str = "streamer"
) -> Path:
    """
    Membuat file subtitle .ass dengan styling tepat sesuai spesifikasi:
    - Mode Streamer: Subtitle ucapan diletakkan di batas bawah facecam (y ≈ 585 pada kanvas 1920)
    - Mode Biasa: Subtitle ucapan di bagian bawah kanvas (MarginV = 160)
    - Kata tampil per kelompok 3-4 kata dinamis (sentence case)
    """
    # Kumpulkan kata-kata yang berada di rentang klip
    clip_words = []
    for seg in words_data:
        for w in seg.get("words", []):
            if w["start"] >= (clip_start - 0.2) and w["end"] <= (clip_end + 0.5):
                clip_words.append(w)

    # Header format ASS
    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
"""
    # MarginV dihitung dari bawah layar:
    # Pada Mode Streamer: Target y ≈ 585 dari atas. Karena canvas 1920, MarginV = 1920 - 585 = 1335.
    if mode == "streamer":
        style_line = "Style: Default,Montserrat ExtraBold,36,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4.0,1,2,40,40,1335,1\n"
    else:
        # Mode Biasa: Di bawah tengah, aman dari UI TikTok
        style_line = "Style: Default,Montserrat ExtraBold,40,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4.2,1,2,40,40,220,1\n"

    events_header = "\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"

    dialogue_lines = []

    # Kelompokkan kata menjadi kelompok 3 - 4 kata
    chunk_size = 4
    for i in range(0, len(clip_words), chunk_size):
        chunk = clip_words[i : i + chunk_size]
        if not chunk:
            continue
        
        # Waktu relatif terhadap awal klip
        rel_start = max(0.0, chunk[0]["start"] - clip_start)
        rel_end = max(rel_start + 0.4, chunk[-1]["end"] - clip_start)

        start_str = format_ass_time(rel_start)
        end_str = format_ass_time(rel_end)

        text_str = " ".join([w["word"] for w in chunk]).strip()
        if text_str:
            dialogue_lines.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text_str}")

    content = ass_header + style_line + events_header + "\n".join(dialogue_lines)
    output_ass_path.write_text(content, encoding="utf-8")
    return output_ass_path


# ==============================================================================
# STEP 7: RENDERING BATCH (FFMPEG)
# ==============================================================================
def get_ffmpeg_font_path(font_filename: str) -> str:
    """Mencari path file font (.ttf) di folder fonts/ proyek atau C:/Windows/Fonts lalu meng-escape untuk FFmpeg."""
    candidates = [
        FONTS_DIR / font_filename,
        Path("C:/Windows/Fonts") / font_filename,
        FONTS_DIR / "Anton-Regular.ttf",
        FONTS_DIR / "Poppins-Bold.ttf",
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for cand in candidates:
        if cand.exists():
            escaped = str(cand.resolve()).replace("\\", "/")
            return escaped.replace(":", "\\:")
    return ""


def split_hook_into_two_lines(raw_text: str) -> List[str]:
    """Memecah teks headline hook menjadi maksimal 2 baris yang seimbang panjang karakternya."""
    cleaned = raw_text.strip()
    if not cleaned:
        return [""]

    # Jika pengguna sudah memisahkan dengan baris baru (\n)
    if "\n" in cleaned:
        parts = [p.strip() for p in cleaned.splitlines() if p.strip()]
        if len(parts) <= 2:
            return parts
        return [parts[0], " ".join(parts[1:])]

    words = cleaned.split()
    if len(words) <= 4 and len(cleaned) <= 26:
        return [cleaned]

    # Cari titik potong kata yang menghasilkan selisih panjang karakter paling seimbang
    best_idx = len(words) // 2
    best_diff = 9999
    for i in range(1, len(words)):
        l1 = " ".join(words[:i])
        l2 = " ".join(words[i:])
        diff = abs(len(l1) - len(l2))
        if diff < best_diff:
            best_diff = diff
            best_idx = i

    return [" ".join(words[:best_idx]), " ".join(words[best_idx:])]


def render_clip_ffmpeg(
    source_video: Path,
    clip: ClipSegment,
    clip_index: int,
    channel_name: str,
    mode: str,
    facecam_box: Tuple[int, int, int, int],
    orig_w: int,
    orig_h: int,
    words_data: List[Dict[str, Any]],
    output_dir: Path,
    temp_dir: Path,
    font_style: str = "anton",
    total_clips: int = MAX_CLIPS
) -> Path:
    """
    Merender 1 klip video vertikal 1080x1920 menggunakan filter_complex FFmpeg.
    """
    clip_filename = f"clip_{clip_index:02d}.mp4"
    final_output = output_dir / clip_filename
    sub_file = temp_dir / f"sub_{clip_index:02d}.ass"

    start_sec = clip.start_seconds
    end_sec = clip.end_seconds
    duration = end_sec - start_sec

    # 1. Buat file subtitle ASS untuk segmen ini
    generate_ass_subtitle_file(words_data, start_sec, end_sec, sub_file, mode=mode)

    # 2. Path escaping untuk FFmpeg di Windows
    # Gunakan forward slash dan escape titik dua (C\:/path/to/...)
    sub_escaped = str(sub_file.resolve()).replace("\\", "/").replace(":", "\\:")
    fonts_dir_escaped = str(FONTS_DIR.resolve()).replace("\\", "/").replace(":", "\\:")

    # 3. Font & Teks untuk CC: YouTube (Poppins Bold 30px) dan Judul Hook (#FFD700 Kuning Emas)
    preset = FONT_PRESETS.get((font_style or "anton").lower(), FONT_PRESETS["anton"])
    cc_font_escaped = get_ffmpeg_font_path("Poppins-Bold.ttf")
    hook_font_escaped = get_ffmpeg_font_path(preset["file"])

    channel_clean = escape_ffmpeg_text(channel_name)

    raw_hook_text = clip.hook_headline.upper() if preset.get("uppercase") else clip.hook_headline
    raw_hook_lines = split_hook_into_two_lines(raw_hook_text)
    max_line_chars = max((len(line) for line in raw_hook_lines), default=1)

    base_size = preset["base_size"]
    max_normal = preset["max_chars_normal"]
    if max_line_chars > max_normal:
        hook_fontsize = max(30, int(base_size * (max_normal / max_line_chars)))
    else:
        hook_fontsize = base_size

    escaped_hook_lines = [escape_ffmpeg_text(line) for line in raw_hook_lines]

    cc_font_opt = f"fontfile='{cc_font_escaped}':" if cc_font_escaped else ""
    hook_font_opt = f"fontfile='{hook_font_escaped}':" if hook_font_escaped else ""
    sub_filter = f"subtitles='{sub_escaped}':fontsdir='{fonts_dir_escaped}'" if FONTS_DIR.exists() else f"subtitles='{sub_escaped}'"

    print(f"\n[Rendering {clip_index:02d}/{total_clips:02d}] -> {clip_filename} ({duration:.1f}s) | Font: {preset['label']}")
    print(f"  Judul: {clip.title}")

    if mode == "streamer":
        # MODE 1: STREAMER (Facecam 1080x640 + Gameplay 1080x960 + Hook 1080x320)
        fc_x, fc_y, fc_w, fc_h = facecam_box

        # Gameplay crop rasio 1080:960 = 1.125
        gp_w = int(orig_h * (1080.0 / 960.0))
        if gp_w > orig_w:
            gp_w = orig_w
        # Center crop horizontal gameplay
        gp_x = max(0, (orig_w - gp_w) // 2)

        if len(escaped_hook_lines) >= 2:
            hook_drawtext = (
                f"[layer4]drawtext={hook_font_opt}text='{escaped_hook_lines[0]}':fontsize={hook_fontsize}:fontcolor=#FFD700:"
                f"borderw=4:bordercolor=black:shadowcolor=black@0.9:shadowx=3:shadowy=3:x=(w-text_w)/2:y=1640[layer4b];"
                f"[layer4b]drawtext={hook_font_opt}text='{escaped_hook_lines[1]}':fontsize={hook_fontsize}:fontcolor=#FFD700:"
                f"borderw=4:bordercolor=black:shadowcolor=black@0.9:shadowx=3:shadowy=3:x=(w-text_w)/2:y=1715[layer5];"
            )
        else:
            hook_drawtext = (
                f"[layer4]drawtext={hook_font_opt}text='{escaped_hook_lines[0]}':fontsize={hook_fontsize}:fontcolor=#FFD700:"
                f"borderw=4:bordercolor=black:shadowcolor=black@0.9:shadowx=3:shadowy=3:x=(w-text_w)/2:y=1675[layer5];"
            )

        filter_complex = (
            # Kanvas dasar hitam 1080x1920
            f"color=c=black:s=1080x1920:d={duration}[bg];"
            # Facecam: crop dari koordinat terdeteksi & scale ke 1080x640
            f"[0:v]crop={fc_w}:{fc_h}:{fc_x}:{fc_y},scale=1080:640:force_original_aspect_ratio=increase,crop=1080:640[fc];"
            # Gameplay: crop tengah & scale ke 1080x960
            f"[0:v]crop={gp_w}:{orig_h}:{gp_x}:0,scale=1080:960:force_original_aspect_ratio=increase,crop=1080:960[gp];"
            # Gabungkan facecam (y=0) dan gameplay (y=640)
            f"[bg][fc]overlay=0:0[layer1];"
            f"[layer1][gp]overlay=0:640[layer2];"
            # Garis batas halus antar zona
            f"[layer2]drawbox=y=638:w=1080:h=3:color=white@0.2:t=fill[layer3];"
            # Credit Sumber pojok kiri atas (Bold & Jelas)
            f"[layer3]drawtext={cc_font_opt}text='CC\\: YouTube / {channel_clean}':fontsize=30:fontcolor=white@0.95:x=32:y=30:"
            f"box=1:boxcolor=black@0.68:boxborderw=14:borderw=1:bordercolor=black@0.4[layer4];"
            # Headline Hook Kuning Emas (#FFD700) rata tengah per baris di zona bawah
            f"{hook_drawtext}"
            # Bakar Subtitle ASS ucapan
            f"[layer5]{sub_filter}[v_out]"
        )

    else:
        # MODE 2: BIASA (Center Crop 9:16 Layar Penuh)
        crop_w = int(orig_h * (9.0 / 16.0))
        crop_x = (orig_w - crop_w) // 2

        filter_complex = (
            f"[0:v]crop={crop_w}:{orig_h}:{crop_x}:0,scale=1080:1920[v_scaled];"
            f"[v_scaled]drawtext={cc_font_opt}text='CC\\: YouTube / {channel_clean}':fontsize=30:fontcolor=white@0.95:x=32:y=40:"
            f"box=1:boxcolor=black@0.68:boxborderw=14:borderw=1:bordercolor=black@0.4[v_cc];"
            f"[v_cc]{sub_filter}[v_out]"
        )

    # Coba gunakan akselerasi NVIDIA NVENC jika mode GPU aktif
    vcodec = "h264_nvenc" if WHISPER_DEVICE == "cuda" else "libx264"
    codec_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "22"] if vcodec == "h264_nvenc" else ["-c:v", "libx264", "-preset", "fast", "-crf", "21"]

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-t", str(duration),
        "-i", str(source_video),
        "-filter_complex", filter_complex,
        "-map", "[v_out]",
        "-map", "0:a:0?"
    ] + codec_args + [
        "-c:a", "aac",
        "-b:a", "192k",
        str(final_output)
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"[OK] Selesai: {final_output.name} ({final_output.stat().st_size / (1024*1024):.2f} MB)")
    except subprocess.CalledProcessError as err:
        if vcodec == "h264_nvenc":
            # Fallback ke libx264 CPU encoder
            print(f"[INFO] NVENC gagal, beralih ke CPU encoder libx264...")
            fallback_cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_sec),
                "-t", str(duration),
                "-i", str(source_video),
                "-filter_complex", filter_complex,
                "-map", "[v_out]",
                "-map", "0:a:0?",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "21",
                "-c:a", "aac",
                "-b:a", "192k",
                str(final_output)
            ]
            subprocess.run(fallback_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            print(f"[OK] Selesai: {final_output.name} ({final_output.stat().st_size / (1024*1024):.2f} MB)")
        else:
            err_msg = err.stderr.decode("utf-8", errors="ignore")
            print(f"[ERROR] Gagal merender {clip_filename}: {err_msg[-400:]}")
            raise err

    return final_output


# ==============================================================================
# MAIN CONTROLLER PIPELINE
# ==============================================================================
def main():
    # 1. Parsing argumen CLI
    parser = argparse.ArgumentParser(description="Auto YouTube Clipper (Windows Edition)")
    parser.add_argument("url", nargs="?", default="", help="Tautan/Link YouTube")
    parser.add_argument("-m", "--mode", choices=["streamer", "biasa", "1", "2"], default="", help="Mode layout: 'streamer' (1) atau 'biasa' (2)")
    parser.add_argument("-f", "--facecam", choices=["auto", "bottom-right", "bottom-left", "top-right", "top-left"], default="auto", help="Posisi webcam streamer (default: auto)")
    parser.add_argument("--font-style", choices=list(FONT_PRESETS.keys()), default="anton", help="Gaya font judul hook Kuning Emas (default: anton)")
    parser.add_argument("--clips", type=int, default=MAX_CLIPS, help=f"Maksimal klip yang dirender (default: {MAX_CLIPS})")

    args, unknown = parser.parse_known_args()

    # Tangani argumen kedua sebagai mode jika ada yang memasukkan: python auto_clipper.py <url> <mode>
    if not args.mode and len(unknown) > 0:
        if unknown[0] in ["streamer", "biasa", "1", "2"]:
            args.mode = unknown[0]

    print("=" * 70)
    print("       AUTO YOUTUBE CLIPPER - WINDOWS EDITION (9:16)")
    print("=" * 70)

    # 2. Cek Dependensi
    check_dependencies()

    # Siapkan direktori kerja
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Input Parameter / Interaktif
    youtube_url = args.url
    if not youtube_url:
        print("\n--- [LANGKAH 1] INPUT PENGGUNA ---")
        youtube_url = input("Masukkan Tautan/Link YouTube: ").strip()

    if not youtube_url:
        print("[ERROR] Link YouTube tidak boleh kosong.")
        return

    mode_raw = args.mode
    if not mode_raw:
        print("\nPilih Mode Layout:")
        print("  1. Mode Streamer (Layout Split Atas-Bawah + Facecam + Safe Zone)")
        print("  2. Mode Biasa (Center Crop 9:16 Full Screen)")
        mode_raw = input("Pilihan Anda [1 / 2] (Default: 1): ").strip()

    mode = "biasa" if mode_raw in ["2", "biasa"] else "streamer"
    print(f"\n[OK] URL Target : {youtube_url}")
    print(f"[OK] Mode Layout: {'Mode Streamer (Facecam + Gameplay)' if mode == 'streamer' else 'Mode Biasa (Center Crop 9:16)'}")

    temp_video = TEMP_DIR / "temp_source.mp4"

    try:
        # 3. Ekstraksi Metadata Channel & Unduh Video
        channel_name = get_channel_name(youtube_url)
        download_youtube_video(youtube_url, temp_video)

        # Ukuran dan durasi video
        orig_w, orig_h, duration = get_video_dimensions(temp_video)
        print(f"[OK] Resolusi Video: {orig_w}x{orig_h}, Durasi: {duration:.1f} detik ({duration/60:.1f} menit)")

        # 4. Deteksi Wajah (Khusus Mode Streamer)
        facecam_box = (0, 0, orig_w, orig_h)
        if mode == "streamer":
            facecam_box = detect_streamer_facecam(temp_video, orig_w, orig_h, facecam_pos=args.facecam)

        # 5. Transkripsi Full Audio Lokal
        transcription_data = transcribe_audio_with_whisper(temp_video, total_duration=duration)
        if not transcription_data:
            print("[ERROR] Transkripsi audio kosong atau tidak ditemukan suara ucapan.")
            return

        # 6. Pemilihan Momen & Hook via Gemini AI
        clips = analyze_moments_with_gemini(transcription_data, duration)
        if not clips:
            print("[ERROR] Tidak ada momen klip yang berhasil dipilih oleh Gemini.")
            return

        # 7. Rendering Batch FFmpeg
        print(f"\n--- [LANGKAH 2] MEMULAI PROSES RENDERING ({len(clips)} KLIP) ---")
        rendered_clips = []
        for idx, clip in enumerate(clips, 1):
            try:
                out_clip = render_clip_ffmpeg(
                    source_video=temp_video,
                    clip=clip,
                    clip_index=idx,
                    channel_name=channel_name,
                    mode=mode,
                    facecam_box=facecam_box,
                    orig_w=orig_w,
                    orig_h=orig_h,
                    words_data=transcription_data,
                    output_dir=OUTPUT_DIR,
                    temp_dir=TEMP_DIR,
                    font_style=args.font_style,
                    total_clips=len(clips)
                )
                rendered_clips.append((idx, clip, out_clip))
            except Exception as e:
                print(f"[WARN] Melewatkan klip ke-{idx} karena error: {e}")

        # 8. Simpan Ringkasan Informasi Klip
        info_file = OUTPUT_DIR / "info_klip.txt"
        summary_lines = [
            f"INFORMASI HASIL CLIPPING YOUTUBE",
            f"Channel Sumber : {channel_name}",
            f"Mode Layout    : {mode.capitalize()}",
            f"Total Klip     : {len(rendered_clips)} klip",
            f"Tanggal Proses : {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            ""
        ]
        for idx, clip, path in rendered_clips:
            summary_lines.append(f"Klip {idx:02d}: {path.name}")
            summary_lines.append(f"  Judul SEO : {clip.title}")
            summary_lines.append(f"  Durasi    : {clip.start_seconds}s s/d {clip.end_seconds}s ({(clip.end_seconds - clip.start_seconds):.1f}s)")
            summary_lines.append(f"  Hook Teks : \"{clip.hook_headline}\"")
            summary_lines.append(f"  Alasan    : {clip.reason}")
            summary_lines.append("-" * 60)

        info_file.write_text("\n".join(summary_lines), encoding="utf-8")
        print(f"\n[OK] Ringkasan informasi klip disimpan ke: {info_file}")

        print("\n" + "=" * 70)
        print("PROSES SELESAI DENGAN SUKSES!")
        print(f"Semua klip vertikal tersimpan di folder: {OUTPUT_DIR.resolve()}")
        print("=" * 70)

    finally:
        # 9. Pembersihan File Sementara
        print("\nMembersihkan file sementara...")
        if temp_video.exists():
            try:
                temp_video.unlink()
                print("[OK] temp_source.mp4 dihapus.")
            except Exception:
                pass
        
        # Bersihkan file sub .ass sementara
        for sub_f in TEMP_DIR.glob("sub_*.ass"):
            try:
                sub_f.unlink()
            except Exception:
                pass
        
        try:
            TEMP_DIR.rmdir()
        except Exception:
            pass


if __name__ == "__main__":
    main()
