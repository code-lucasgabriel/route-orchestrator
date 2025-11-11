#!/bin/bash
# quick_analysis.sh
# Quick start script for performance analysis

set -e  # Exit on error

echo "=========================================="
echo "HFFVRPTW Performance Analysis Quick Start"
echo "=========================================="
echo ""

# Set Python to use local venv with python3.14
VENV_PATH="venv/bin/python3.14"
VENV_PIP="venv/bin/pip"

# Check if venv exists, if not create it
if [ ! -d "venv" ]; then
    echo "Creating virtual environment with python3.14..."
    python3.14 -m venv venv
    echo "✓ Virtual environment created"
fi

# Use venv python
PYTHON="$VENV_PATH"
PIP="$VENV_PIP"

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Python 3.14 venv not found at $PYTHON"
    echo "Please create venv with: python3.14 -m venv venv"
    exit 1
fi

echo "✓ Using Python: $($PYTHON --version) from venv"
echo ""

# Check if logs exist
if [ ! -d "logs" ]; then
    echo "ERROR: logs/ directory not found."
    echo "Please run experiments first with: python main.py"
    exit 1
fi

LOG_COUNT=$(find logs -name "*.txt" -path "*/results/*" | wc -l)
echo "✓ Found $LOG_COUNT result files in logs/"
echo ""

# Install dependencies
echo "Step 1: Installing dependencies..."
echo "------------------------------------"
$PIP install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Test parser
echo "Step 2: Testing parser..."
echo "-------------------------"
$PYTHON test_parser.py
echo ""

# Parse logs
echo "Step 3: Parsing logs..."
echo "-----------------------"
$PYTHON parser.py
echo ""

# Generate plots
echo "Step 4: Generating plots..."
echo "---------------------------"
$PYTHON plotter.py
echo ""

# Summary
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  • results.parquet       - Parsed data ($(du -h results.parquet 2>/dev/null | cut -f1))"
echo "  • plots/                - Visualizations"
echo ""
echo "Next steps:"
echo "  1. Open plots/ directory to view figures"
echo "  2. Import summary_*.csv into your report"
echo "  3. Run custom analysis with results.parquet"
echo ""
echo "For more information:"
echo "  • PLOTTING_README.md    - Detailed guide"
echo "  • PLOTTING_SUMMARY.md   - Feature overview"
echo "  • WORKFLOW.md           - Visual workflow"
echo ""
