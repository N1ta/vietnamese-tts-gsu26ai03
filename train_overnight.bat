@echo off
setlocal
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set "PROJ=D:\GSU26AI03 - SU26"
set "PYF=C:\Users\Nita\miniconda3\envs\f5_env\python.exe"
set "PYC=C:\Users\Nita\miniconda3\envs\coqui\python.exe"
set "RUN=%PROJ%\models\fastspeech2_punct\runs\fastspeech2_infore-June-28-2026_09+01PM-0000000"
set "CFG=%PROJ%\models\fastspeech2_punct\coqui_config.json"
cd /d "%PROJ%"

echo ==================================================
echo  QUA DEM: sinh MEN bai_dai  -^>  train FS2
echo ==================================================

echo [1/2] Sinh MEN bai_dai (best-of-3, ~25 phut)...
"%PYF%" gen_baidai_robust.py --ref dataset/MEN_dataset/wavs/0006.wav --ref-text-file demo/refs/men_reftext.txt --out demo/f5_men/bai_dai.wav --speed 1.1 --nfe 48 --k 3
if errorlevel 1 echo [!] MEN bai_dai gap loi - van train FS2 tiep.

echo.
echo [2/2] Train FastSpeech2 (restore tu checkpoint_37176). Sang mai bam Ctrl+C de dung.
"%PYC%" -m TTS.bin.train_tts --restore_path "%RUN%\checkpoint_37176.pth" --config_path "%CFG%"

echo.
echo === DA DUNG/XONG. Checkpoint FS2 luu trong run folder moi (moi 5000 step). ===
pause
