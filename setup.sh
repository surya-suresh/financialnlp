#!/bin/bash
set -e

if command -v module &>/dev/null; then
    module load python/3.12
fi

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

pip install --upgrade pip --quiet
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118 --quiet
pip install -r requirements.txt
