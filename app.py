"""
Auto YouTube Clipper - Web UI Server (FastAPI + Tailwind CSS + Vue 3)
======================================================================
Menyediakan antarmuka Web Studio interaktif di browser lokal dengan
animasi progress bar real-time (0 - 100%), live preview player 9:16,
dan fitur kustomisasi momen AI sebelum render.
"""

import os
import sys
import json
import time
import asyncio
import threading
import webbrowser
from pathlib import Path
from typing import List, Dict, Any, Optional

# Otomatis install FastAPI & Uvicorn jika belum terpasang
try:
    import fastapi
    import uvicorn
except ImportError:
    print("[INFO] Memasang dependensi web server (fastapi, uvicorn)...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "fastapi>=0.100.0", "uvicorn>=0.23.0"], check=True)
    import fastapi
    import uvicorn

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import fungsi-fungsi dari auto_clipper.py
from auto_clipper import (
    BASE_DIR,
    OUTPUT_DIR,
    TEMP_DIR,
    WHISPER_DEVICE,
    WHISPER_MODEL_SIZE,
    GEMINI_MODEL_NAME,
    get_channel_name,
    download_youtube_video,
    get_video_dimensions,
    detect_streamer_facecam,
    transcribe_audio_with_whisper,
    analyze_moments_with_gemini,
    render_clip_ffmpeg,
    ClipSegment,
    setup_cuda_dlls,
    setup_yt_dlp
)

setup_cuda_dlls()
setup_yt_dlp()

app = FastAPI(title="Auto YouTube Clipper AI Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMPLATES_DIR = BASE_DIR / "templates"
INDEX_HTML = TEMPLATES_DIR / "index.html"

# Global state untuk sesi web saat ini
current_task = {
    "is_running": False,
    "source_video": None,
    "channel_name": "Creator",
    "mode": "streamer",
    "font_style": "anton",
    "facecam_box": None,
    "orig_w": 1920,
    "orig_h": 1080,
    "duration": 0.0,
    "words_data": [],
    "candidate_clips": []
}

# ==============================================================================
# WEBSOCKET CONNECTION MANAGER & REAL-TIME LOG BROADCAST
# ==============================================================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[INFO] Browser terhubung ke WebSocket. Total klien aktif: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[INFO] Browser terputus dari WebSocket. Sisa klien: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()
loop: Optional[asyncio.AbstractEventLoop] = None

latest_progress_state = {
    "step": 0,
    "overallPercent": 0,
    "stepPercent": 0,
    "statusMessage": "Siap memproses video",
    "downloadText": "",
    "facecamText": "",
    "whisperText": "",
    "geminiText": "",
    "renderText": ""
}


# Event bus berbasis HTTP polling (tidak bergantung pada WebSocket / event loop)
events_lock = threading.Lock()
events_buffer: List[dict] = []
event_counter = 0
MAX_EVENTS = 4000


def send_ws(msg: dict):
    """Simpan event ke buffer; browser mengambilnya lewat /api/events (thread-safe)."""
    global latest_progress_state, event_counter
    if msg.get("type") == "progress":
        if "step" in msg:
            latest_progress_state["step"] = msg["step"]
        if "overallPercent" in msg:
            latest_progress_state["overallPercent"] = msg["overallPercent"]
        if "stepPercent" in msg:
            latest_progress_state["stepPercent"] = msg["stepPercent"]
        if "statusMessage" in msg:
            latest_progress_state["statusMessage"] = msg["statusMessage"]

        step = msg.get("step")
        detail = msg.get("detail", "")
        if step == 1 and detail:
            latest_progress_state["downloadText"] = detail
        elif step == 2 and detail:
            latest_progress_state["facecamText"] = detail
        elif step == 3 and detail:
            latest_progress_state["whisperText"] = detail
        elif step == 4 and detail:
            latest_progress_state["geminiText"] = detail
        elif step == 5 and detail:
            latest_progress_state["renderText"] = detail

    with events_lock:
        event_counter += 1
        item = dict(msg)
        item["id"] = event_counter
        events_buffer.append(item)
        if len(events_buffer) > MAX_EVENTS:
            del events_buffer[: len(events_buffer) - MAX_EVENTS]


def log_ws(text: str, level: str = "info"):
    """Kirim log ke terminal fisik dan langsung broadcast ke antarmuka web browser."""
    print(text)


class BroadcastStream:
    """Mengalirkan seluruh output sys.stdout / sys.stderr (termasuk dari yt-dlp & ffmpeg) ke browser."""
    def __init__(self, original_stream, level="info"):
        self.original_stream = original_stream
        self.level = level
        self.last_send = 0.0

    def write(self, text: str):
        if self.original_stream:
            self.original_stream.write(text)
            self.original_stream.flush()

        if not text:
            return

        clean = text.replace("\r", "\n")
        lines = [l.strip() for l in clean.split("\n") if l.strip()]
        now = time.time()
        for line in lines:
            if line.startswith("\x1b"):
                continue

            # Throttle baris download progress agar websocket tetap ringan & lancar
            if line.startswith("[download]") and "%" in line:
                if now - self.last_send < 0.2:
                    continue
                self.last_send = now

            send_ws({"type": "log", "message": line, "level": self.level})

    def flush(self):
        if self.original_stream:
            self.original_stream.flush()

    def fileno(self):
        return getattr(self.original_stream, "fileno", lambda: 1)()

    def isatty(self):
        return getattr(self.original_stream, "isatty", lambda: False)()


# Pasang hook output sistem agar seluruh log terminal tampil di browser secara langsung
sys.stdout = BroadcastStream(sys.stdout, level="info")
sys.stderr = BroadcastStream(sys.stderr, level="error")


@app.on_event("startup")
async def startup_event():
    global loop
    loop = asyncio.get_running_loop()

    # Abaikan error tidak berbahaya di Windows saat browser menutup koneksi
    # (refresh halaman, preview video dibatalkan, tab ditutup, dll).
    default_handler = loop.get_exception_handler()

    def quiet_handler(lp, context):
        exc = context.get("exception")
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        if default_handler:
            default_handler(lp, context)
        else:
            lp.default_exception_handler(context)

    loop.set_exception_handler(quiet_handler)
    print("[OK] Web Studio siap. Server aktif.")


# ==============================================================================
# REQUEST MODELS
# ==============================================================================
class StartRequest(BaseModel):
    url: str
    mode: str = "streamer"
    facecam: str = "bottom-right"
    font_style: str = "anton"
    action: str = "auto"  # 'auto' atau 'review'


class RenderRequest(BaseModel):
    clips: List[Dict[str, Any]]
    font_style: Optional[str] = None


# ==============================================================================
# BACKGROUND WORKER PIPELINE
# ==============================================================================
def background_pipeline_worker(req: StartRequest):
    global current_task
    current_task["is_running"] = True
    temp_video = TEMP_DIR / "temp_source.mp4"
    current_task["source_video"] = temp_video
    current_task["mode"] = req.mode
    current_task["font_style"] = req.font_style or "anton"

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        TEMP_DIR.mkdir(parents=True, exist_ok=True)

        # STEP 1: Unduh Video YouTube
        send_ws({
            "type": "progress",
            "step": 1,
            "overallPercent": 5,
            "stepPercent": 0,
            "statusMessage": "Menghubungkan ke server YouTube dan membaca metadata...",
            "detail": "Menghubungkan..."
        })
        log_ws(f"Mengekstrak nama channel dari URL: {req.url}")
        channel_name = get_channel_name(req.url)
        current_task["channel_name"] = channel_name

        log_ws("Memulai pengunduhan video dengan progress real-time...")
        
        dl_state = {"max": 0.0}

        def on_dl_progress(d):
            status = d.get("status", "downloading")
            raw_p = d.get("percent", 0.0)
            # Persentase dibuat monoton (video & audio diunduh bergantian) dan
            # ditahan di 99% sampai ekstraksi/penggabungan selesai.
            if status == "downloading":
                dl_state["max"] = max(dl_state["max"], raw_p)
            p = round(min(dl_state["max"] if status == "downloading" else 99.0, 99.0), 1)
            overall = int(5 + (p * 0.25))  # 5% s/d 30%
            sp = d.get("speed", "")
            eta = d.get("eta", "")
            dl_mb = d.get("downloaded_mb", "")
            tot_mb = d.get("total_mb", "")
            custom_msg = d.get("status_text")
            filename = d.get("filename", "")

            if custom_msg:
                msg = custom_msg
                detail = "Ekstraksi / Merging..."
            elif status == "finished":
                msg = f"Format {filename} selesai diunduh. Menggabungkan format media..."
                detail = "Ekstraksi..."
            elif sp:
                msg = f"Mengunduh: {raw_p}% ({dl_mb}/{tot_mb}) • {sp} • {eta}"
                detail = f"{sp} {eta}".strip()
            else:
                msg = f"Mengunduh video: {raw_p}%"
                detail = "Mengunduh..."

            send_ws({
                "type": "progress",
                "step": 1,
                "overallPercent": overall,
                "stepPercent": p,
                "statusMessage": msg,
                "detail": detail
            })

        download_youtube_video(req.url, temp_video, progress_callback=on_dl_progress)

        orig_w, orig_h, duration = get_video_dimensions(temp_video)
        current_task["orig_w"] = orig_w
        current_task["orig_h"] = orig_h
        current_task["duration"] = duration
        log_ws(f"Video terunduh: {orig_w}x{orig_h}, Durasi: {duration:.1f} detik ({duration/60:.1f} menit)")

        send_ws({
            "type": "progress",
            "step": 1,
            "overallPercent": 30,
            "stepPercent": 100,
            "statusMessage": f"Video 1080p berhasil diunduh ({duration/60:.1f} menit)",
            "detail": "100% Selesai"
        })

        # STEP 2: Deteksi Facecam
        send_ws({
            "type": "progress",
            "step": 2,
            "overallPercent": 40,
            "stepPercent": 50,
            "statusMessage": "Memindai posisi area webcam streamer...",
            "detail": "Memindai..."
        })
        if req.mode == "streamer":
            fc_box = detect_streamer_facecam(temp_video, orig_w, orig_h, facecam_pos=req.facecam)
            current_task["facecam_box"] = fc_box
            log_ws(f"Area webcam ditetapkan: {fc_box}")
        else:
            current_task["facecam_box"] = (0, 0, orig_w, orig_h)

        send_ws({
            "type": "progress",
            "step": 2,
            "overallPercent": 50,
            "stepPercent": 100,
            "statusMessage": "Tata letak facecam siap.",
            "detail": "100% Siap"
        })

        # STEP 3: Transkripsi Whisper Lokal (CUDA)
        send_ws({
            "type": "progress",
            "step": 3,
            "overallPercent": 50,
            "stepPercent": 0,
            "statusMessage": f"Memulai transkripsi suara lokal dengan GPU CUDA (Whisper {WHISPER_MODEL_SIZE})...",
            "detail": "0% Memulai..."
        })
        log_ws("Memulai proses transkripsi suara lokal dengan pelacakan progress real-time...")

        def on_whisper_progress(d):
            p = d.get("percent", 0.0)
            overall = int(50 + (p * 0.25))  # 50% s/d 75%
            cur = d.get("current_sec", 0.0)
            tot = d.get("total_sec", 0.0)
            cur_fmt = f"{int(cur//60):02d}:{int(cur%60):02d}"
            tot_fmt = f"{int(tot//60):02d}:{int(tot%60):02d}"
            txt = d.get("text", "")
            snippet = f' • "{txt[:35]}..."' if txt else ""
            send_ws({
                "type": "progress",
                "step": 3,
                "overallPercent": overall,
                "stepPercent": p,
                "statusMessage": f"Transkripsi audio GPU: {p}% ({cur_fmt}/{tot_fmt}){snippet}",
                "detail": f"{p}% ({cur_fmt})"
            })

        words_data = transcribe_audio_with_whisper(temp_video, total_duration=duration, progress_callback=on_whisper_progress)
        current_task["words_data"] = words_data

        send_ws({
            "type": "progress",
            "step": 3,
            "overallPercent": 75,
            "stepPercent": 100,
            "statusMessage": f"Transkripsi selesai ({len(words_data)} segmen kalimat)",
            "detail": f"100% ({len(words_data)} segmen)"
        })

        # STEP 4: Kurasi Momen AI (Gemini 3.8 Flash)
        send_ws({
            "type": "progress",
            "step": 4,
            "overallPercent": 85,
            "stepPercent": 0,
            "statusMessage": "Gemini 3.8 Flash sedang memilih momen paling viral...",
            "detail": "Menganalisis..."
        })
        log_ws(f"Menganalisis momen terbaik dengan Gemini ({GEMINI_MODEL_NAME})...")
        clips = analyze_moments_with_gemini(words_data, duration)
        current_task["candidate_clips"] = clips

        send_ws({
            "type": "progress",
            "step": 4,
            "overallPercent": 90,
            "stepPercent": 100,
            "statusMessage": f"Gemini berhasil memilih {len(clips)} momen terbaik!",
            "detail": f"{len(clips)} Klip Siap"
        })

        # Jika mode review, kirim daftar momen ke browser dan tunggu konfirmasi pengguna
        if req.action == "review":
            clips_dict = [c.model_dump() for c in clips]
            send_ws({
                "type": "candidates",
                "clips": clips_dict,
                "statusMessage": "Silakan tinjau dan kustomisasi momen klip di bawah sebelum dirender."
            })
            log_ws("Momen klip dikirim ke browser untuk ditinjau oleh pengguna.")
            current_task["is_running"] = False
            return

        # Jika mode auto, langsung lanjutkan ke STEP 5 (Rendering)
        execute_rendering_pipeline(clips)

    except Exception as e:
        log_ws(f"Terjadi kesalahan di pipeline: {e}", level="error")
        send_ws({"type": "error", "message": str(e)})
        current_task["is_running"] = False


def execute_rendering_pipeline(clips_to_render: List[Any]):
    """Merender klip-klip yang disetujui pengguna secara batch."""
    global current_task
    source_video = current_task["source_video"]
    channel_name = current_task["channel_name"]
    mode = current_task["mode"]
    facecam_box = current_task["facecam_box"]
    orig_w = current_task["orig_w"]
    orig_h = current_task["orig_h"]
    words_data = current_task["words_data"]
    font_style = current_task.get("font_style", "anton")

    total = len(clips_to_render)
    log_ws(f"\n--- MEMULAI BATCH RENDERING ({total} KLIP) ---")

    rendered_list = []

    for idx, c_item in enumerate(clips_to_render, 1):
        if isinstance(c_item, dict):
            clip_obj = ClipSegment(**c_item)
        else:
            clip_obj = c_item

        current_percent = int(75 + ((idx - 1) / total) * 25)
        step_p = int(((idx - 1) / total) * 100)
        send_ws({
            "type": "progress",
            "step": 5,
            "overallPercent": current_percent,
            "stepPercent": step_p,
            "statusMessage": f"Merender klip {idx:02d}/{total:02d}: '{clip_obj.title}'",
            "detail": f"Klip {idx}/{total} sedang dirender..."
        })

        out_path = render_clip_ffmpeg(
            source_video=source_video,
            clip=clip_obj,
            clip_index=idx,
            channel_name=channel_name,
            mode=mode,
            facecam_box=facecam_box,
            orig_w=orig_w,
            orig_h=orig_h,
            words_data=words_data,
            output_dir=OUTPUT_DIR,
            temp_dir=TEMP_DIR,
            font_style=font_style,
            total_clips=total
        )

        rendered_list.append((idx, clip_obj, out_path))

        done_p = int((idx / total) * 100)
        send_ws({
            "type": "progress",
            "step": 5,
            "overallPercent": int(75 + (idx / total) * 25),
            "stepPercent": done_p,
            "statusMessage": f"Klip {idx:02d}/{total:02d} selesai dirender",
            "detail": f"{idx}/{total} klip selesai"
        })

        # Beritahu browser bahwa klip ini sudah selesai dan bisa langsung diputar
        send_ws({
            "type": "clip_completed",
            "clip": {
                "index": idx,
                "filename": out_path.name,
                "title": clip_obj.title,
                "hook_headline": clip_obj.hook_headline,
                "duration": f"{(clip_obj.end_seconds - clip_obj.start_seconds):.1f}s"
            }
        })

    # Simpan ringkasan info_klip.txt
    info_file = OUTPUT_DIR / "info_klip.txt"
    summary_lines = [
        f"INFORMASI HASIL CLIPPING YOUTUBE",
        f"Channel Sumber : {channel_name}",
        f"Mode Layout    : {mode.capitalize()}",
        f"Total Klip     : {len(rendered_list)} klip",
        f"Tanggal Proses : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        ""
    ]
    for idx, clip, path in rendered_list:
        summary_lines.append(f"Klip {idx:02d}: {path.name}")
        summary_lines.append(f"  Judul SEO : {clip.title}")
        summary_lines.append(f"  Durasi    : {clip.start_seconds}s s/d {clip.end_seconds}s ({(clip.end_seconds - clip.start_seconds):.1f}s)")
        summary_lines.append(f"  Hook Teks : \"{clip.hook_headline}\"")
        summary_lines.append(f"  Alasan    : {clip.reason}")
        summary_lines.append("-" * 60)

    info_file.write_text("\n".join(summary_lines), encoding="utf-8")

    send_ws({
        "type": "finished",
        "totalClips": len(rendered_list)
    })
    log_ws(f"Selesai! Seluruh {len(rendered_list)} klip berhasil dibuat.")
    current_task["is_running"] = False


# ==============================================================================
# FASTAPI ENDPOINTS
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="Template index.html tidak ditemukan")
    return HTMLResponse(content=INDEX_HTML.read_text(encoding="utf-8"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global loop
    if loop is None or not loop.is_running():
        try:
            loop = asyncio.get_running_loop()
        except Exception:
            pass

    await manager.connect(websocket)
    
    # Kirim sinkronisasi state progress terkini ke klien web
    try:
        await websocket.send_json({
            "type": "sync_state",
            "is_running": current_task["is_running"],
            "progress": latest_progress_state
        })
    except Exception:
        pass

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/events")
def get_events(since: int = 0):
    """Polling: kembalikan event (log, progress, klip) yang lebih baru dari id `since`."""
    with events_lock:
        new_events = [e for e in events_buffer if e["id"] > since]
        last_id = event_counter
    return {
        "events": new_events,
        "last_id": last_id,
        "is_running": current_task["is_running"],
        "progress": latest_progress_state,
    }


@app.post("/api/start")
async def start_pipeline(req: StartRequest):
    if current_task["is_running"]:
        raise HTTPException(status_code=400, detail="Proses sedang berjalan. Tunggu hingga selesai.")

    # Reset state progress & buffer event untuk proses baru
    with events_lock:
        events_buffer.clear()
    latest_progress_state.update({
        "step": 1, "overallPercent": 5, "stepPercent": 0,
        "statusMessage": "Menghubungkan ke YouTube...",
        "downloadText": "", "facecamText": "", "whisperText": "",
        "geminiText": "", "renderText": ""
    })
    current_task["is_running"] = True

    threading.Thread(target=background_pipeline_worker, args=(req,), daemon=True).start()
    return {"status": "started", "action": req.action}


@app.post("/api/render")
async def render_clips(req: RenderRequest):
    if not req.clips:
        raise HTTPException(status_code=400, detail="Daftar klip kosong")

    if req.font_style:
        current_task["font_style"] = req.font_style
    current_task["is_running"] = True
    threading.Thread(target=execute_rendering_pipeline, args=(req.clips,), daemon=True).start()
    return {"status": "rendering_started", "count": len(req.clips)}


@app.get("/api/clips")
async def get_existing_clips():
    """Mengembalikan daftar klip yang sudah pernah dirender di output_clips."""
    clips = []
    if OUTPUT_DIR.exists():
        for f in sorted(OUTPUT_DIR.glob("clip_*.mp4")):
            clips.append({
                "filename": f.name,
                "title": f.stem.replace("_", " ").title(),
                "hook_headline": "Siap unggah untuk YouTube Shorts & TikTok",
                "duration": f"{f.stat().st_size / (1024*1024):.1f} MB"
            })
    return {"clips": clips}


@app.get("/api/video/{filename}")
async def stream_video(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File video tidak ditemukan")
    return FileResponse(path=file_path, media_type="video/mp4", filename=filename)


@app.post("/api/open-folder")
async def open_folder():
    """Membuka folder output_clips di Windows Explorer."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(str(OUTPUT_DIR.resolve()))
    return {"status": "opened"}


# ==============================================================================
# LAUNCHER
# ==============================================================================
def open_browser():
    time.sleep(1.2)
    print("\n[OK] Membuka antarmuka Web Studio di browser default...")
    webbrowser.open("http://127.0.0.1:8000")


if __name__ == "__main__":
    if sys.platform == "win32":
        # SelectorEventLoop lebih stabil daripada ProactorEventLoop untuk koneksi lokal
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    threading.Thread(target=open_browser, daemon=True).start()

    print("\n" + "=" * 70)
    print("   AUTO YOUTUBE CLIPPER - WEB STUDIO (FASTAPI + VUE 3 + TAILWIND)")
    print("=" * 70)
    print("Server berjalan di: http://127.0.0.1:8000")
    print("Tekan Ctrl+C di terminal ini untuk mematikan server.")
    print("=" * 70 + "\n")

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
