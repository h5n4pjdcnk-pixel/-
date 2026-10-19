@echo off
title Energy Storage RAG Web UI
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found in PATH.
    pause
    exit /b 1
)

echo ============================================================
echo   Starting the Streamlit web application...
echo   URL: http://localhost:8501
echo   Press Ctrl+C or close this window to stop.
echo ============================================================
echo.

python -m streamlit run app.py

echo.
echo The web application has stopped.
pause