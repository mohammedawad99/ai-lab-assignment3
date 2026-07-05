#!/bin/sh
# Run the Assignment 3 entry point; defaults to showing help.
cd "$(dirname "$0")" || exit 1
if [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
else
    PY=python3
fi
if [ $# -eq 0 ]; then
    exec "$PY" a3.py --help
fi
exec "$PY" a3.py "$@"
