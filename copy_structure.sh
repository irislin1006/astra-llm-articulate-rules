#!/bin/bash
# Copy essential files from llm-articulate-rules to astra-llm-articulate-rules

SOURCE="/home/iris/wsl_shared/llm-articulate-rules"
DEST="/home/iris/wsl_shared/astra-llm-articulate-rules"

echo "Copying research project files..."

# Create directory structure
mkdir -p "$DEST/data"
mkdir -p "$DEST/results"
mkdir -p "$DEST/src/llm_clients"
mkdir -p "$DEST/src/utils"
mkdir -p "$DEST/scripts"
mkdir -p "$DEST/reports"

# Copy data files
echo "Copying data files..."
cp "$SOURCE"/data/*.json "$DEST/data/" 2>/dev/null || echo "No JSON data files"

# Copy all result files
echo "Copying result files..."
cp "$SOURCE"/results/*.json "$DEST/results/" 2>/dev/null || echo "No result files"

# Copy source code
echo "Copying source code..."
cp "$SOURCE"/src/__init__.py "$DEST/src/" 2>/dev/null
cp "$SOURCE"/src/llm_clients/*.py "$DEST/src/llm_clients/" 2>/dev/null
cp "$SOURCE"/src/utils/*.py "$DEST/src/utils/" 2>/dev/null

# Copy main scripts
echo "Copying main scripts..."
cp "$SOURCE"/run_step1_classification.py "$DEST/" 2>/dev/null
cp "$SOURCE"/run_step2_articulation.py "$DEST/" 2>/dev/null
cp "$SOURCE"/run_step3_articulation_faithfulness_simple.py "$DEST/" 2>/dev/null
cp "$SOURCE"/run_step3_articulation_faithfulness.py "$DEST/" 2>/dev/null
cp "$SOURCE"/run_ablation_few_shot.py "$DEST/" 2>/dev/null
cp "$SOURCE"/run_probe_position_bias.py "$DEST/" 2>/dev/null
cp "$SOURCE"/run_probe_sycophancy.py "$DEST/" 2>/dev/null

# Copy reports
echo "Copying reports..."
cp "$SOURCE"/REPORT_GPT41_FINAL.md "$DEST/reports/" 2>/dev/null
cp "$SOURCE"/REPORT_GPT41MINI.md "$DEST/reports/" 2>/dev/null
cp "$SOURCE"/llm_articulation_report.tex "$DEST/" 2>/dev/null

# Copy configuration files
echo "Copying configuration files..."
cp "$SOURCE"/requirements.txt "$DEST/" 2>/dev/null
cp "$SOURCE"/.env.example "$DEST/" 2>/dev/null || echo ".env.example not found (optional)"

echo "Copy complete!"
