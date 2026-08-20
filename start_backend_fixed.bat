@echo off
setlocal
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set "PROJ=D:\GSU26AI03 - SU26"
set "PYF=C:\Users\Nita\miniconda3\envs\f5_env\python.exe"
set "PYC=C:\Users\Nita\miniconda3\envs\coqui\python.exe"
cd /d "%PROJ%"

echo ==================================================
echo  VietVoice BACKEND (2 backend: F5 + Coqui) + ngrok
echo  Frontend vietvoice-gsu1.vercel.app se goi vao day
echo ==================================================
echo [1] Backend Coqui (VITS/FS2/Tacotron2) o cong 7861...
powershell -NoProfile -Command "$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'; Start-Process -FilePath '%PYC%' -ArgumentList 'api_coqui.py' -WorkingDirectory '%PROJ%'"

echo [2] Backend F5 (Nu/Nam, GPU) o cong 7860 (tai model ~1 phut)...
powershell -NoProfile -Command "$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'; Start-Process -FilePath '%PYF%' -ArgumentList 'api_f5.py' -WorkingDirectory '%PROJ%'"

echo.
echo [3] Backend cong khai (co dinh): https://strangely-satirical-onstage.ngrok-free.dev
echo     (mo link nay se thay {"status":"ready", ... 5 models} khi F5 tai xong)
echo.
"%PROJ%\ngrok.exe" http 7860 --url=https://strangely-satirical-onstage.ngrok-free.dev

echo.
echo === Da dung. Dong 2 cua so de tat. ===
pause
