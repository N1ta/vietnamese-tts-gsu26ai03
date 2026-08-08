@echo off
setlocal
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set "PROJ=D:\GSU26AI03 - SU26"
set "PYC=C:\Users\Nita\miniconda3\envs\coqui\python.exe"
set "RUN=%PROJ%\models\fastspeech2_punct\runs\fastspeech2_infore-July-25-2026_05+19AM-0000000"
cd /d "%PROJ%"

echo Train FastSpeech2 - RESUME tu checkpoint_45000 (July run, continue_path)...
echo (giu nguyen step + optimizer + LR; de chay den ~150k roi Ctrl+C)
"%PYC%" -m TTS.bin.train_tts --continue_path "%RUN%"

echo.
echo === DA DUNG/XONG. Checkpoint luu trong run folder June-28. ===
pause
