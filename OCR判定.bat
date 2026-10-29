@echo off
rem Inspection report OCR: start the GUI (double-click this file)
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
  echo uv was not found. Please install uv: https://docs.astral.sh/uv/
  pause
  exit /b 1
)
rem Create or update the Python environment (.venv) if needed
uv sync --quiet
if errorlevel 1 (
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" ocr_gui.py
