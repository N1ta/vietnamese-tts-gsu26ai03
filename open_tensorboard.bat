@echo off
setlocal
set "PROJ=D:\GSU26AI03 - SU26"
set "PY=C:\Users\Nita\miniconda3\envs\coqui\python.exe"
cd /d "%PROJ%"

echo ============================================================
echo  TensorBoard - 4 mo hinh trong 1 bang (de so sanh)
echo    VITS / Tacotron2 / FastSpeech2 / HiFi-GAN
echo  Dia chi: http://localhost:6006
echo ============================================================
echo Dang khoi dong TensorBoard... GIU cua so nay mo.
echo Neu trinh duyet bao loi luc dau -> cho ~5 giay roi bam Refresh (F5).
echo.

start "" http://localhost:6006

"%PY%" -m tensorboard.main --logdir_spec VITS:models\vits_punct\runs,VITS-fair:models\vits_fair\runs,Tacotron2:models\tacotron2\runs,FastSpeech2:models\fastspeech2_punct\runs,HiFiGAN:models\hifigan\runs --port 6006

echo.
echo === TensorBoard da dung. Dong cua so de tat. ===
pause
