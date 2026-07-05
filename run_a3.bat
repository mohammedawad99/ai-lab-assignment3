@echo off
rem Run the Assignment 3 entry point; defaults to showing help.
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
    set PY=.venv\Scripts\python.exe
) else (
    set PY=python
)
if "%~1"=="" (
    %PY% a3.py --help
) else (
    %PY% a3.py %*
)
exit /b %errorlevel%
