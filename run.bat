@echo off
title NutriKids Server
echo =========================================
echo      Menjalankan NutriKids Server
echo =========================================
echo.

REM Cek dan aktifkan virtual environment
if exist "venv311\Scripts\activate.bat" (
    echo [*] Mengaktifkan virtual environment venv311...
    call "venv311\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    echo [*] Mengaktifkan virtual environment venv...
    call "venv\Scripts\activate.bat"
) else (
    echo [!] Peringatan: Virtual environment tidak ditemukan di folder venv/venv311.
    echo [*] Pastikan Python sudah terinstall.
)

echo.
echo [*] Memeriksa dependencies (requirements.txt)...
pip install -r requirements.txt

echo.
echo [*] Menjalankan Server...
echo [*] Aplikasi akan bisa diakses di: http://127.0.0.1:5000
echo =========================================
echo Tekan CTRL+C di terminal ini untuk mematikan server.
echo.

python app.py

echo.
pause
