@echo off
echo ========================================
echo   Notes Q/A Bot - Windows Setup
echo ========================================
echo.

echo [1/4] Checking Docker...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Docker Engine is not running!
    echo.
    echo Please:
    echo   1. Open Docker Desktop from your Start Menu or taskbar
    echo   2. Wait until it shows "Engine running" (green dot, bottom-left)
    echo   3. Run this setup.bat again
    echo.
    echo See fix_docker.md for detailed instructions.
    pause
    exit /b 1
)
echo Docker is running!
echo.

echo [2/4] Starting Endee Vector Database...
docker compose up -d
echo.

echo [3/4] Setting up .env file...
if not exist .env (
    copy .env.example .env
    echo .env file created!
    echo.
    echo IMPORTANT: Open .env and set your GROQ_API_KEY
    echo Get a FREE key at: https://console.groq.com/
) else (
    echo .env already exists.
)
echo.

echo [4/4] Installing Python packages...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo Package install failed. Trying with --upgrade...
    pip install --upgrade -r requirements.txt
)
echo.

echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Open .env and add your GROQ_API_KEY
echo      (Free key at https://console.groq.com/)
echo   2. Run: streamlit run app.py
echo.
pause
