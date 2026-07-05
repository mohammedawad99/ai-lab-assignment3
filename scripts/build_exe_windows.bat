@echo off
rem Optional: build a one-file Windows executable a3.exe with PyInstaller.
rem PyInstaller is a packaging-only tool and is NOT in requirements.txt;
rem install it separately. Build outputs (dist\, build\, *.spec) are
rem generated artifacts and must not be committed.
cd /d "%~dp0\.."
if exist .venv\Scripts\python.exe (
    set PY=.venv\Scripts\python.exe
) else (
    set PY=python
)
%PY% -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo PyInstaller is not installed. Install it with:
    echo   %PY% -m pip install pyinstaller
    exit /b 1
)
%PY% -m PyInstaller --onefile --name a3 a3.py
echo built: dist\a3.exe (generated artifact, do not commit)
