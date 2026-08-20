@echo off
setlocal
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set "PROJ=D:\GSU26AI03 - SU26"
set "PY=C:\Users\Nita\miniconda3\envs\coqui\python.exe"
cd /d "%PROJ%"

echo ==================================================
echo  TRAIN VITS-fair (~60.9M) - capacity matched
echo  Config : models\vits_fair\config.json
echo  Output : models\vits_fair\runs\vits_fair-<ngay>\
echo ==================================================
echo LUU Y QUAN TRONG:
echo   - TAT backend (dong cua so start_backend_fixed.bat) TRUOC khi train.
echo     F5 va train VITS khong chay chung 6GB GPU duoc (se OOM).
echo   - Xem 50-100 buoc dau: co dong "avg_loss" / loss giam la OK -> treo may.
echo   - Dung train: dong cua so nay, hoac Ctrl+C.
echo ==================================================
echo.

"%PY%" -m TTS.bin.train_tts --config_path models\vits_fair\config.json

echo.
echo === Train da dung. Checkpoint o models\vits_fair\runs\ ===
pause
