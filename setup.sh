#!/bin/bash
# setup.sh — One-time setup script for the Stanford Daily research project

set -e

echo "=========================================="
echo "Stanford Daily Research Project Setup"
echo "=========================================="
echo ""

# Check Python version
echo "[1/4] Checking Python version …"
python3 --version || { echo "ERROR: Python 3 not found. Install Python 3.9+"; exit 1; }

# Create virtual environment
echo ""
echo "[2/4] Creating virtual environment …"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate and install dependencies
echo ""
echo "[3/4] Installing dependencies …"
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy model
echo ""
echo "[4/4] Downloading spaCy model …"
python -m spacy download en_core_web_sm

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate       # Linux/Mac"
echo "  venv\\Scripts\\activate          # Windows"
echo ""
echo "To run the full pipeline (from the project root):"
echo "  python -m src.run_pipeline"
echo ""
echo "To run individual modules (from the project root):"
echo "  python -m src.collect"
echo "  python -m src.clean"
echo "  python -m src.quality"
echo "  etc."
echo ""
echo "NOTE: always use 'python -m src.<module>', never 'python src/<module>.py'."
echo "      The modules use absolute 'src.' imports and require package context."
echo ""
