@echo off
title Auto YouTube Clipper (Windows Edition)
cd /d "%~dp0"

echo ========================================================
echo        AUTO YOUTUBE CLIPPER - WINDOWS LAUNCHER
echo ========================================================
echo.

:: Cek file .env
if not exist ".env" (
    echo [INFO] File .env belum ada. Mengcopy dari .env.example...
    copy ".env.example" ".env" >nul
    echo [PENTING] Silakan isi GEMINI_API_KEY di file .env sebelum melanjutkan!
    echo.
)

:: Cek keberadaan Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python tidak terdeteksi di PATH!
    echo Pastikan Python 3.10+ sudah terpasang.
    pause
    exit /b 1
)

:: Jalankan skrip clipper
python auto_clipper.py %*

echo.
echo ========================================================
pause
