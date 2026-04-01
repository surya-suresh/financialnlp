#!/bin/bash
# setup.sh — Create virtual environment and install dependencies
# Works on both OSC Pitzer and local machines.

set -e

echo "=================================================="
echo "  Finance NLP Pipeline — Environment Setup"
echo "=================================================="

# Load Python module if on OSC (safe no-op elsewhere)
if command -v module &>/dev/null; then
    echo "Detected HPC module system — finding available Python …"
    # Try common OSC Pitzer module names in order of preference
    LOADED=0
    for MOD in python/3.11.4 python/3.10.8 python/3.9.12 python/3.9 python/3.10 python/3.11; do
        if module load "$MOD" 2>/dev/null; then
            echo "Loaded module: $MOD"
            LOADED=1
            break
        fi
    done
    if [ "$LOADED" -eq 0 ]; then
        echo "No versioned python module found — trying 'python' …"
        module load python 2>/dev/null || echo "Module load skipped; using system Python."
    fi
fi

# Create venv if it doesn't already exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment …"
    python3 -m venv venv
else
    echo "Virtual environment already exists, reusing."
fi

source venv/bin/activate

echo "Upgrading pip …"
pip install --upgrade pip --quiet

# Install PyTorch with CUDA 11.8 wheels FIRST.
# This ensures sm_70 (V100, CC 7.0) kernels are included.
# PyTorch 2.4+ dropped sm_70 support, so we pin to 2.3.1.
echo "Installing PyTorch 2.3.1 (cu118, V100-compatible) …"
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118 --quiet

echo "Installing remaining dependencies …"
pip install -r requirements.txt

echo ""
echo "Setup complete.  Activate with:  source venv/bin/activate"
echo "Then run:                        python src/main.py"
