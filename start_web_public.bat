@echo off
setlocal
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set "PROJ=D:\GSU26AI03 - SU26"
set "PYF=C:\Users\Nita\miniconda3\envs\f5_env\python.exe"
cd /d "%PROJ%"

echo ==================================================
echo  VietVoice - Web + link cong khai (cloudflared)
echo ==================================================
echo [1] Khoi dong web VietVoice o cua so rieng (tai model ~1 phut)...
powershell -NoProfile -Command "$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'; Start-Process -FilePath '%PYF%' -ArgumentList 'web_f5.py' -WorkingDirectory '%PROJ%'"

echo [2] Tao link cong khai... doi web tai xong thi link se hoat dong.
echo     TIM dong  https://....trycloudflare.com  o ben duoi:
echo.
"%PROJ%\cloudflared.exe" tunnel --url http://localhost:7860

echo.
echo === Da dung. Dong ca 2 cua so de tat web. ===
pause
