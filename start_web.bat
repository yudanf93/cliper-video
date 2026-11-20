@echo off
title Auto YouTube Clipper Web Studio
cd /d "%~dp0"

echo ========================================================
echo      AUTO YOUTUBE CLIPPER - WEB STUDIO LAUNCHER
echo ========================================================
echo.
echo Memeriksa dependensi dan menyalakan Web Studio di http://127.0.0.1:8000 ...
echo Halaman browser akan terbuka otomatis.
echo.

set "PY_CMD=python"
where python >nul 2>nul
if %errorlevel% neq 0 (
    if exist "D:\laragon\bin\python\python-3.10\python.exe" (
        set "PY_CMD=D:\laragon\bin\python\python-3.10\python.exe"
    ) else (
        echo [ERROR] Python tidak ditemukan di PATH maupun di D:\laragon\bin\python\python-3.10!
        pause
        exit /b 1
    )
)

%PY_CMD% app.py

pause
