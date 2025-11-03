"""Runner for Probe B: Sycophancy Test

Tests if models follow suggestions ("I think the answer is X") and rationalize them,
even when suggestions are wrong.

Usage:
    python run_probe_sycophancy.py --models "gpt-4o" "claude-3-5-sonnet-20241022"
    python run_probe_sycophancy.py --provider openai --test-zeroshot
    python run_probe_sycophancy.py  # Test all models from config
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.experiments.step3_probe_sycophancy import SycophancyProbe
from src.utils.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_datasets(data_dir: str = "data") -> dict:
    """Load all generated datasets."""
    data_path = Path(data_dir)
    datasets = {}

    for file in data_path.glob("*.json"):
        with open(file, 'r') as f:
            dataset = json.load(f)
            datasets[dataset["task_id"]] = dataset

    logger.info(f"Loaded {len(datasets)} datasets")
    return datasets


def main():
    parser = argparse.ArgumentParser(description="Run sycophancy probe (suggestion following test)")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to test (e.g., 'gpt-4o' 'claude-3-5-sonnet-20241022')"
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "openrouter"],
        help="Test all models from a specific provider"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing datasets"
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--test-zeroshot",
        action="store_true",
        help="Also test zero-shot vs few-shot susceptibility"
    )

    args = parser.parse_args()

    # Load config
    config = Config(args.config)

    # Determine which models to test
    if args.models:
        models = args.models
    elif args.provider:
        models = config.get_models_by_provider(args.provider)
    else:
        models = config.get_all_models()

    logger.info(f"Testing models: {models}")

    # Load datasets
    datasets = load_datasets(args.data_dir)
    if not datasets:
        logger.error("No datasets found! Run generate_data.py first")
        return

    # Run probe
    probe = SycophancyProbe(config, results_dir=args.results_dir)

    results = probe.run_all_conditions(
        models=models,
        datasets=datasets,
        test_zeroshot=args.test_zeroshot,
        save_results=True
    )

    # Print summary
    print("\n" + "="*80)
    print("SYCOPHANCY PROBE SUMMARY")
    print("="*80)

    for model, model_data in results.items():
        print(f"\nModel: {model}")
        print("-" * 80)

        if model_data.get("comparisons"):
            summary = probe.get_summary_stats(model_data["comparisons"])
            print(f"  Tasks tested: {summary['num_tasks']}")
            print(f"  Mean wrong suggestion follow rate: {summary['mean_wrong_follow_rate']:.2%}")
            print(f"  Max wrong suggestion follow rate: {summary['max_wrong_follow_rate']:.2%}")
            print(f"  High sycophancy tasks: {summary['num_high_sycophancy']}/{summary['num_tasks']} ({summary['high_sycophancy_rate']:.1%})")

            # Show worst cases
            comparisons = model_data["comparisons"]
            worst = sorted(comparisons, key=lambda x: x["sycophancy_effect"], reverse=True)[:3]
            print(f"\n  Most sycophantic tasks:")
            for c in worst:
                print(f"    - {c['task_id']}: {c['sycophancy_effect']:.1%} wrong follow rate, {c['interpretation']}")

    print("\n" + "="*80)
    print(f"Results saved to {args.results_dir}/")
    print("="*80)


if __name__ == "__main__":
    main()
