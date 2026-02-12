#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$DIR/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv --system-site-packages "$DIR/venv"
    "$DIR/venv/bin/pip" install -q -r "$DIR/requirements.txt"
fi

exec "$DIR/venv/bin/python" "$DIR/lstt.py" "$@"
