#!/bin/sh
# Optional: build a one-file Unix executable `a3` with PyInstaller.
# PyInstaller is a packaging-only tool and is NOT in requirements.txt;
# install it separately. Build outputs (dist/, build/, *.spec) are
# generated artifacts and must not be committed.
cd "$(dirname "$0")/.." || exit 1
if [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
else
    PY=python3
fi
if ! "$PY" -m PyInstaller --version >/dev/null 2>&1; then
    echo "PyInstaller is not installed. Install it with:"
    echo "  $PY -m pip install -r requirements-build.txt"
    exit 1
fi
"$PY" -m PyInstaller --onefile --name a3 --console a3.py
echo "built: dist/a3 (generated artifact, do not commit)"
