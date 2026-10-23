@echo off
cd /d "%~dp0"
echo ============================================
echo   XAI-NIDS Platform (Web Version)
echo ============================================
echo.
pip install fastapi uvicorn -q
echo Starting server...
timeout /t 2 /nobreak >nul
start http://127.0.0.1:8000
python server.py
pause
