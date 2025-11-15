#!/bin/bash

# run_pipeline.sh
# 
# Complete HFFVRPTW analysis pipeline:
# 1. Run experiments (main.py)
# 2. Parse logs (parser.py)
# 3. Generate core visualizations (plotter.py)
# 4. Generate advanced analysis (analysis_suite.py)

set -e  # Exit on any error

echo "=============================================================================="
echo "HFFVRPTW ANALYSIS PIPELINE"
echo "=============================================================================="
echo ""

# Step 1: Run experiments
echo "STEP 1/4: Running experiments (main.py)"
echo "------------------------------------------------------------------------------"
python3 main.py
echo ""
echo "✓ Experiments complete"
echo ""

# Step 2: Parse logs
echo "STEP 2/4: Parsing logs (parser.py)"
echo "------------------------------------------------------------------------------"
python3 parser.py
echo ""
echo "✓ Parsing complete"
echo ""

# Step 3: Generate core visualizations
echo "STEP 3/4: Generating core visualizations (plotter.py)"
echo "------------------------------------------------------------------------------"
python3 plotter.py
echo ""
echo "✓ Core visualizations complete"
echo ""

# Step 4: Generate advanced analysis
echo "STEP 4/4: Generating advanced analysis (analysis_suite.py)"
echo "------------------------------------------------------------------------------"
python3 analysis_suite.py
echo ""
echo "✓ Advanced analysis complete"
echo ""

# Summary
echo "=============================================================================="
echo "PIPELINE COMPLETE"
echo "=============================================================================="
echo ""
echo "Generated files:"
echo "  - results.parquet (944 records)"
echo "  - plots/images/*.png (26 visualization files)"
echo "  - plots/PDFs/*.pdf (26 vector graphics)"
echo "  - plots/summaries/*.csv (4 summary tables)"
echo ""
echo "Next steps:"
echo "  - Review plots in plots/images/"
echo "  - Check summaries in plots/summaries/"
echo "  - Use results.parquet for custom analysis"
echo ""
