# Reproduction Guide

This guide provides step-by-step instructions to reproduce all experimental results.

## Prerequisites

1. **Python 3.8+** installed
2. **OpenAI API key** with access to GPT-4.1 models
3. Approximately **$50-100 in API credits** (depending on tasks run)
4. **~8 hours** total runtime for full reproduction

## Setup

```bash
# 1. Clone repository
git clone https://github.com/irislin1006/astra-llm-articulate-rules.git
cd astra-llm-articulate-rules

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API key
export OPENAI_API_KEY="sk-..."  # Replace with your key
```

## Full Reproduction (All Experiments)

### Step 1: Classification Baseline (Estimated: 2 hours)

```bash
# Run 10-shot classification on all 10 tasks for both models
python run_step1_classification.py \
  --models "gpt-4.1-2025-04-14,gpt-4.1-mini-2025-04-14" \
  --shots 10

# Expected output: results/step1_gpt-4.1-2025-04-14_*.json
#                  results/step1_gpt-4.1-mini-2025-04-14_*.json
```

**Expected Results:**
- GPT-4.1: 89.4% average accuracy
- GPT-4.1-Mini: 86.0% average accuracy

### Step 2: Articulation Test (Estimated: 30 minutes)

```bash
# Test rule articulation for both models
python run_step2_articulation.py \
  --models "gpt-4.1-2025-04-14,gpt-4.1-mini-2025-04-14"

# Expected output: results/step2_gpt-4.1-2025-04-14_*.json
#                  results/step2_gpt-4.1-mini-2025-04-14_*.json
```

**Expected Results:**
- Both models: 100% (10/10 tasks correct)

### Step 3a: Direct Articulation-Faithfulness (Estimated: 2 hours)

**This is the critical experiment that reveals the paradox.**

```bash
# Run direct faithfulness test
python run_step3_articulation_faithfulness_simple.py \
  --models "gpt-4.1-2025-04-14"

python run_step3_articulation_faithfulness_simple.py \
  --models "gpt-4.1-mini-2025-04-14"

# Expected output: results/step3_articulation_faithfulness_*.json
#                  results/step3_articulation_comparison_*.json
```

**Expected Results:**
- GPT-4.1: 50% faithful tasks (5/10)
- GPT-4.1-Mini: 20% faithful tasks (2/10)
- even_digit_sum catastrophe: 96% → 40% (-56%)

### Step 3b: Position Bias Probe (Estimated: 1.5 hours)

```bash
# Test for position bias shortcuts
python run_probe_position_bias.py \
  --models "gpt-4.1-2025-04-14,gpt-4.1-mini-2025-04-14"

# Expected output: results/probe_position_bias_*.json
```

**Expected Results:**
- Both models: 30% unfaithful tasks (3/10)
- Mean drop: GPT-4.1: 8.2%, GPT-4.1-Mini: 11.4%

### Step 3c: Sycophancy Probe (Estimated: 1 hour)

```bash
# Test resistance to wrong suggestions
python run_probe_sycophancy.py \
  --models "gpt-4.1-2025-04-14,gpt-4.1-mini-2025-04-14"

# Expected output: results/probe_sycophancy_*.json
```

**Expected Results:**
- Both models: 20% moderate sycophancy (2/10)

### Ablation Study: Few-Shot Learning Curves (Estimated: 3 hours)

```bash
# Test with 3, 5, 8, 10 examples
python run_ablation_few_shot.py \
  --models "gpt-4.1-2025-04-14,gpt-4.1-mini-2025-04-14" \
  --shots "3,5,8,10"

# Expected output: results/ablation_few_shot_*.json
```

**Expected Results:**
- GPT-4.1: Non-monotonic curve (5-shot dip)
- GPT-4.1-Mini: Monotonic improvement

## Quick Validation (Subset)

To validate the key findings quickly (~1 hour, ~$10):

```bash
# Run only on GPT-4.1, subset of tasks
python run_step1_classification.py \
  --models "gpt-4.1-2025-04-14" \
  --shots 10 \
  --tasks "all_lowercase,even_digit_sum,contains_exclamation"

python run_step2_articulation.py \
  --models "gpt-4.1-2025-04-14" \
  --tasks "all_lowercase,even_digit_sum,contains_exclamation"

python run_step3_articulation_faithfulness_simple.py \
  --models "gpt-4.1-2025-04-14"
```

## Verifying Results

### Check Output Files

```bash
# List all result files
ls -lh results/

# Check Step 1 results
python -c "
import json
with open('results/step1_gpt-4.1-2025-04-14_XXXXXX.json') as f:
    data = json.load(f)
    print(f'Tasks: {len(data)}')
    print(f'Avg accuracy: {sum(d[\"accuracy\"] for d in data)/len(data):.2%}')
"

# Check Step 3 comparison
python -c "
import json
with open('results/step3_articulation_comparison_gpt-4.1-2025-04-14_XXXXXX.json') as f:
    data = json.load(f)
    faithful = sum(1 for d in data if 'FAITHFUL' in d['interpretation'] and 'UNDER' not in d['interpretation'] and 'OVER' not in d['interpretation'])
    print(f'Faithful tasks: {faithful}/{len(data)}')
"
```

## Analyzing Results

The repository includes pre-computed results. To analyze:

```python
import json
import pandas as pd

# Load Step 3 comparison
with open('results/step3_articulation_comparison_gpt-4.1-2025-04-14_20251103_094032.json') as f:
    comparison = json.load(f)

df = pd.DataFrame(comparison)
print(df[['task_id', 'rule_based_accuracy', 'baseline_accuracy', 'accuracy_diff', 'interpretation']])

# Identify catastrophic failures
catastrophic = df[df['accuracy_diff'] < -0.30]
print(f"\nCatastrophic failures:\n{catastrophic}")
```

## Common Issues

### API Rate Limits
If you hit rate limits:
```bash
# Add delays between calls (modify scripts)
# Or run experiments sequentially instead of in parallel
```

### Memory Issues
```bash
# Run one model at a time
python run_step1_classification.py --models "gpt-4.1-2025-04-14"  # Then Mini separately
```

### Missing Results
```bash
# Check if experiments completed
grep "accuracy" results/*.json
```

## Expected Costs

- **Step 1 (Classification):** ~$30 (500 API calls × 2 models)
- **Step 2 (Articulation):** ~$5 (10 calls × 2 models)
- **Step 3a (Direct Test):** ~$30 (500 calls × 2 models)
- **Step 3b (Position Bias):** ~$20 (400 calls × 2 models)
- **Step 3c (Sycophancy):** ~$20 (400 calls × 2 models)
- **Ablation Study:** ~$50 (additional shots)

**Total:** ~$150-200 for complete reproduction

## Time Estimates

- **Sequential execution:** ~8 hours
- **Parallel execution:** ~3 hours (if API limits allow)

## Support

If you encounter issues:
1. Check logs in `*.log` files
2. Verify API key is set: `echo $OPENAI_API_KEY`
3. Ensure Python 3.8+: `python --version`
4. Open an issue on GitHub

## Pre-computed Results

This repository includes pre-computed results in `results/`. You can analyze these directly without re-running experiments.

```bash
# Count result files
ls results/*.json | wc -l  # Should be 20+
```

---

Last updated: November 3, 2025
