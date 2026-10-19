@echo off
title Multimodal RGB-Thermal Human Detection Studio
cd /d "%~dp0"
echo ======================================================================
echo Starting Multimodal RGB-Thermal Human Detection Studio...
echo ======================================================================

if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe run_app.py
) else (
    python run_app.py
)

pause
