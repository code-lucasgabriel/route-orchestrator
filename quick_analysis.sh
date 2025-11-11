#!/bin/bash
# quick_analysis.sh
# Quick start script for performance analysis

set -e  # Exit on error

echo "=========================================="
echo "HFFVRPTW Performance Analysis Quick Start"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found. Please install Python 3.8+."
    exit 1
fi

echo "✓ Python found: $(python --version)"
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
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Test parser
echo "Step 2: Testing parser..."
echo "-------------------------"
python test_parser.py
echo ""

# Parse logs
echo "Step 3: Parsing logs..."
echo "-----------------------"
python parser.py
echo ""

# Generate plots
echo "Step 4: Generating plots..."
echo "---------------------------"
python plotter.py
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
