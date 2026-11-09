@echo off
chcp 65001 >nul
rem Run from the project folder: the models and images use relative paths
cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
  echo Python not found. Install Python 3.9+ from python.org and tick "Add python.exe to PATH".
  pause
  exit /b
)

echo Installing / checking libraries...
python -m pip uninstall -y opencv-python-headless >nul 2>&1
python -m pip install --upgrade opencv-python numpy matplotlib
if not exist results mkdir results

:menu
echo.
echo ===== LR-7 OpenCV =====
echo 1 - faces + objects + QR on group.jpg  (result.jpg)
echo 2 - objects on coffee.png               (result_objects.jpg)
echo 3 - QR code on qr_scene.jpg             (result_qr.jpg)
echo 4 - webcam, all modes (Esc or q to quit)
echo 5 - experiments.py (charts in results\)
echo 0 - exit
set /p ch=Choose:
if "%ch%"=="1" python vision_lab7.py --mode all --input images\group.jpg --output result.jpg && start "" result.jpg
if "%ch%"=="2" python vision_lab7.py --mode objects --input images\coffee.png --output result_objects.jpg && start "" result_objects.jpg
if "%ch%"=="3" python vision_lab7.py --mode qr --input images\qr_scene.jpg --output result_qr.jpg && start "" result_qr.jpg
if "%ch%"=="4" python vision_lab7.py --mode all --input 0
if "%ch%"=="5" python experiments.py
if "%ch%"=="0" exit /b
goto menu
