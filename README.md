# LLM Rule Articulation & Faithfulness: The Articulation-Faithfulness Paradox

**Author:** Kuan-yen Lin  
**Email:** iris19132@gmail.com  
**Date:** November 3, 2025

## Overview

This repository contains the complete research code, data, and results for investigating the **Articulation-Faithfulness Paradox** in large language models (GPT-4.1 and GPT-4.1-Mini).

**Key Finding:** Both models achieve 100% articulation accuracy (correctly identifying learned rules) yet exhibit 50-80% unfaithfulness when applying those articulated rules, with catastrophic failures on computational tasks (56% accuracy drop).

## Research Question

*Can LLMs articulate the rules they learn? And if so, do these articulations faithfully explain their classification behavior?*

## Repository Structure

```
.
├── README.md                                    # This file
├── llm_articulation_report.tex                 # Full research report (LaTeX)
├── requirements.txt                             # Python dependencies
│
├── data/                                        # Classification task datasets
│   ├── all_lowercase.json                      # 10 synthetic tasks
│   ├── all_uppercase.json
│   ├── contains_exclamation.json
│   ├── contains_number.json
│   ├── contains_prime.json
│   ├── ends_with_vowel.json
│   ├── even_digit_sum.json
│   ├── even_word_count.json
│   ├── no_repeated_letters.json
│   └── starts_with_vowel.json
│
├── results/                                     # Experimental results
│   ├── step1_*.json                            # Classification results
│   ├── step2_*.json                            # Articulation results
│   ├── step3_articulation_faithfulness_*.json  # Direct faithfulness test
│   ├── step3_articulation_comparison_*.json    # Comparison metrics
│   └── probe_*.json                            # Position bias & sycophancy probes
│
├── reports/                                     # Human-readable reports
│   ├── REPORT_GPT41_FINAL.md                   # GPT-4.1 comprehensive report
│   └── REPORT_GPT41MINI.md                     # GPT-4.1-Mini report
│
├── src/                                         # Source code
│   ├── llm_clients/                            # LLM API clients
│   │   ├── base.py
│   │   ├── openai_client.py
│   │   ├── anthropic_client.py
│   │   └── factory.py
│   └── utils/                                   # Utility functions
│       └── data_utils.py
│
└── scripts/                                     # Main experimental scripts
    ├── run_step1_classification.py             # Step 1: Few-shot classification
    ├── run_step2_articulation.py               # Step 2: Rule articulation
    ├── run_step3_articulation_faithfulness_simple.py  # Step 3: Direct test
    ├── run_ablation_few_shot.py                # Ablation studies
    ├── run_probe_position_bias.py              # Position bias probe
    └── run_probe_sycophancy.py                 # Sycophancy probe
```

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
python run_step1_classification.py --models "gpt-4.1-2025-04-14,gpt-4.1-mini-2025-04-14" --shots 10
```

**Step 2: Articulation (Rule identification)**
```bash
python run_step2_articulation.py --models "gpt-4.1-2025-04-14,gpt-4.1-mini-2025-04-14"
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
python run_ablation_few_shot.py --models "gpt-4.1-2025-04-14" --shots "3,5,8,10"
```

### 3. View Results

Results are saved in `results/` as JSON files. Human-readable reports are in `reports/`.

## Key Results

### Articulation Performance
- **GPT-4.1:** 100% (10/10 tasks)
- **GPT-4.1-Mini:** 100% (10/10 tasks)

Both models perfectly identify learned rules from multiple-choice options.

### Faithfulness Performance

#### Direct Articulation-Faithfulness Test
- **GPT-4.1:** 50% faithful (5/10 tasks)
- **GPT-4.1-Mini:** 20% faithful (2/10 tasks)

#### Catastrophic Failures
- **even_digit_sum:** 96% → 40% (-56% drop) in GPT-4.1
- **even_word_count:** 66% → 34% (-32% drop) in GPT-4.1

#### Position Bias
- **Both models:** 30% unfaithful tasks (3/10)
- **Mean accuracy drop:** GPT-4.1: 8.2%, GPT-4.1-Mini: 11.4%

#### Sycophancy
- **Both models:** 20% moderate sycophancy (2/10)

## The Articulation-Faithfulness Paradox

**Paradox:** Models can perfectly articulate rules they fundamentally cannot apply.

**Implications:**
1. **AI Safety:** Articulation tests alone are insufficient for evaluating reliability
2. **Interpretability:** Verbalized reasoning may not reflect actual decision-making
3. **Alignment:** Aligning stated goals with behavior requires addressing both explicit knowledge and implicit execution

## Dual-Process Hypothesis

Our findings support a dual-process model:
- **System 2 (Explicit):** Articulates rules correctly (100%)
- **System 1 (Implicit):** Struggles to execute multi-step reasoning (40% on computational tasks)

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

## Task Suite

10 synthetic classification tasks spanning multiple difficulty levels:

| Category | Task | Rule Complexity |
|----------|------|-----------------|
| Lexical | all_lowercase | O(n) |
| Lexical | all_uppercase | O(n) |
| Lexical | contains_exclamation | O(n) |
| Lexical | contains_number | O(n) |
| Positional | starts_with_vowel | O(1) |
| Positional | ends_with_vowel | O(1) |
| Counting | even_word_count | O(n) |
| Counting | even_digit_sum | O(n) |
| Complex | contains_prime | O(n log n) |
| Complex | no_repeated_letters | O(n) |

## Methodology

### Three-Stage Evaluation Framework

1. **Classification (Step 1):** Few-shot learning with 3, 5, 8, 10 examples
2. **Articulation (Step 2):** Multiple-choice rule identification
3. **Faithfulness (Step 3):** Three complementary tests:
   - **Direct Test:** Apply articulated rules to classify inputs
   - **Position Bias:** Detect answer-position shortcuts
   - **Sycophancy:** Test resistance to wrong suggestions

## Dependencies

- Python 3.8+
- openai >= 1.0.0
- anthropic (optional, for Claude models)
- tqdm
- numpy
- pandas (for analysis)

See `requirements.txt` for complete list.

## License

This research code is provided for academic and research purposes.

## Contact

**Kuan-yen Lin**  
Email: iris19132@gmail.com  
GitHub: [@irislin1006](https://github.com/irislin1006)

## Acknowledgments

This research was completed as part of the LLM Rule Articulation & Faithfulness Testing framework, investigating fundamental questions about LLM reliability and interpretability.

---

**Time Investment:** 18 hours (within research exercise guidelines)
- Experiment design & implementation: 6 hours
- Experiment execution: 8 hours
- Analysis & report writing: 4 hours
