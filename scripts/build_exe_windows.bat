@echo off
rem Build a one-file Windows executable a3.exe with PyInstaller.
rem Must run on NATIVE Windows (PyInstaller does not cross-compile).
rem PyInstaller is a packaging-only tool, pinned in requirements-build.txt
rem (NOT in the runtime requirements.txt). Build outputs (dist\, build\,
rem *.spec) are generated artifacts and must never be committed.
rem
rem a3.exe is a thin launcher: it delegates subcommands to the repository
rem scripts, so no --add-data bundling is needed - run it from the
rem repository root (or from dist\ inside it) with Python installed.
cd /d "%~dp0\.."
if exist .venv\Scripts\python.exe (
    set PY=.venv\Scripts\python.exe
) else (
    set PY=python
)
%PY% -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo PyInstaller is not installed. Install it with:
    echo   %PY% -m pip install -r requirements-build.txt
    exit /b 1
)
%PY% -m PyInstaller --onefile --name a3 --console a3.py
echo built: dist\a3.exe (generated artifact, do not commit)
