#!/bin/bash
# package.sh — Clean and zip the project for OSC upload.
# Run from the project root directory.

set -e

ZIPNAME="project_osc.zip"

echo "Cleaning temporary files …"
find . -type d -name "__pycache__"       -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "venv"              -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache"     -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true
find . -name ".DS_Store" -delete 2>/dev/null || true

echo "Creating $ZIPNAME …"
zip -r "$ZIPNAME" \
    src/ \
    requirements.txt \
    setup.sh \
    run_job.sh \
    README.md \
    --exclude "*.pyc" \
    --exclude "*/__pycache__/*" \
    --exclude "*/venv/*"

echo ""
echo "Done!  Package: $ZIPNAME"
echo "Upload to OSC and run:  bash setup.sh && sbatch run_job.sh"
