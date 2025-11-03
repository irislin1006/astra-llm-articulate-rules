# LLM Rule Articulation & Faithfulness: The Articulation-Faithfulness Paradox

## Overview

This repository contains the complete research code, data, and results for investigating the **Articulation-Faithfulness Paradox** in large language models (GPT-4.1 and GPT-4.1-Mini).

## Research Question

*Can LLMs articulate the rules they learn? And if so, do these articulations faithfully explain their classification behavior?*

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/irislin1006/astra-llm-articulate-rules.git
cd astra-llm-articulate-rules

# Install dependencies
pip install -r requirements.txt

# Set up OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### 2. Run Experiments

**Step 1: Classification (Few-shot learning)**
```bash
python run_step1.py --models "gpt-4.1-2025-04-14,gpt-4.1-mini-2025-04-14" --shots 10
```

**Step 2: Articulation (Rule identification)**
```bash
python run_step2.py --models "gpt-4.1-2025-04-14,gpt-4.1-mini-2025-04-14"
```

**Step 3: Faithfulness Testing**
```bash
# Direct articulation-faithfulness test (CRITICAL)
python run_step3_articulation_faithfulness_simple.py --models "gpt-4.1-2025-04-14"

# Position bias probe
python run_probe_position_bias.py --models "gpt-4.1-2025-04-14"

# Sycophancy probe
python run_probe_sycophancy.py --models "gpt-4.1-2025-04-14"
```

**Ablation Studies**
```bash
python run_step1.py --models gpt-4.1-2025-04-14 --few-shot-ablation --few-shot-counts 3 5 8
```

### 3. View Results

Results are saved in `results/` as JSON files. Human-readable reports are in `reports/`.

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@techreport{lin2025articulation,
  title={The Articulation-Faithfulness Paradox: A Comparative Study of Rule Learning in GPT-4.1 Models},
  author={Lin, Kuan-yen},
  year={2025},
  institution={LLM Rule Articulation \& Faithfulness Framework}
}
```

## License

This research code is provided for academic and research purposes.

## Contact

**Kuan-yen Lin**  
Email: iris19132@gmail.com  
GitHub: [@irislin1006](https://github.com/irislin1006)

## Acknowledgments

This research was completed with the help of AI and LLM.

---

**Time Investment:** 18 hours (within research exercise guidelines)
- Experiment design & implementation: 6 hours
- Experiment execution: 8 hours
- Analysis & report writing: 4 hours
