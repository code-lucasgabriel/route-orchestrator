#!/bin/bash
# setup_and_plot.sh
# One-time setup and plotting script for HFFVRPTW analysis

set -e  # Exit on error

echo "=========================================="
echo "Setting up Python Environment & Plotting"
echo "=========================================="
echo ""

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment with python3.14..."
    python3.14 -m venv venv
    echo "✓ Virtual environment created"
    echo ""
fi

# Activate venv
source venv/bin/activate

echo "✓ Using Python: $(python --version)"
echo ""

# Install dependencies
echo "Installing dependencies..."
echo "--------------------------"
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Check if results.parquet exists
if [ ! -f "results.parquet" ]; then
    echo "Parsing logs (first time setup)..."
    echo "----------------------------------"
    python parser.py
    echo ""
fi

# Generate plots
echo "Generating plots..."
echo "-------------------"
python plotter.py
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Your plots are in the plots/ directory"
echo ""
echo "To run plotter again later, use:"
echo "  source venv/bin/activate"
echo "  python plotter.py"
echo ""
