@echo off
echo Launching HepaGuard AI FastAPI Backend...
cd /d "%~dp0"
..\..\..\venv\Scripts\python.exe -m uvicorn main:app --reload
pause
