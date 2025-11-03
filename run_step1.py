"""Run Step 1: Classification experiments."""

import logging
import argparse
from src.experiments import ClassificationExperiment
from src.utils import get_config, load_all_datasets

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_fewshot_ablation(experiment, models, datasets, few_shot_counts):
    """
    Run ablation study over different few-shot counts.

    Tests hypothesis: "gains plateau beyond 8 examples"

    Args:
        experiment: ClassificationExperiment instance
        models: List of model names
        datasets: Dict of datasets
        few_shot_counts: List of few-shot counts to test

    Returns:
        Dict with ablation results
    """
    import json
    from datetime import datetime
    from pathlib import Path

    ablation_results = {}

    for model in models:
        logger.info(f"\nRunning few-shot ablation for {model}")
        model_ablation = {}

        for count in few_shot_counts:
            logger.info(f"  Testing with {count} few-shot examples")

            # Temporarily override config
            original_count = experiment.step1_config["num_train_examples"]
            experiment.step1_config["num_train_examples"] = count

            # Run experiment
            model_results = []
            for task_id, dataset in datasets.items():
                try:
                    result = experiment.run_single_task(model, dataset)
                    model_results.append(result)
                except Exception as e:
                    logger.error(f"Error on {task_id}: {e}")

            # Restore original count
            experiment.step1_config["num_train_examples"] = original_count

            # Save results
            safe_model = model.replace("/", "_").replace(":", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)

            filepath = results_dir / f"step1_ablation_{count}shot_{safe_model}_{timestamp}.json"
            with open(filepath, 'w') as f:
                json.dump(model_results, f, indent=2)
            logger.info(f"    Saved to {filepath}")

            # Calculate stats
            accuracies = [r["accuracy"] for r in model_results if "accuracy" in r]
            mean_acc = sum(accuracies) / len(accuracies) if accuracies else 0

            model_ablation[count] = {
                "mean_accuracy": mean_acc,
                "results": model_results
            }

        ablation_results[model] = model_ablation

        # Print ablation summary for this model
        logger.info(f"\n  Few-shot ablation summary for {model}:")
        for count in few_shot_counts:
            acc = model_ablation[count]["mean_accuracy"]
            logger.info(f"    {count} shots: {acc:.2%} accuracy")

    return ablation_results


def main():
    """Run Step 1 experiments."""
    parser = argparse.ArgumentParser(description="Run Step 1: Classification experiments")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Models to test (default: all from config)"
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "openrouter"],
        help="Test only models from specific provider"
    )
    parser.add_argument(
        "--few-shot-ablation",
        action="store_true",
        help="Run ablation over different few-shot counts (3,5,8,10)"
    )
    parser.add_argument(
        "--few-shot-counts",
        nargs="+",
        type=int,
        default=[3, 5, 8, 10],
        help="Few-shot counts for ablation (default: 3 5 8 10)"
    )
    args = parser.parse_args()

    logger.info("Starting Step 1: Classification experiments")

    # Load configuration
    config = get_config()

    # Determine which models to test
    if args.models:
        models = args.models
    elif args.provider:
        models = config.get_models(args.provider)
    else:
        models = config.get_models()

    logger.info(f"Testing {len(models)} models: {models}")

    # Load datasets
    datasets = load_all_datasets("data")
    if not datasets:
        logger.error("No datasets found! Run generate_data.py first.")
        return

    logger.info(f"Loaded {len(datasets)} datasets")

    # Initialize experiment
    experiment = ClassificationExperiment(config)

    # Run experiments
    if args.few_shot_ablation:
        logger.info(f"Running few-shot ablation with counts: {args.few_shot_counts}")
        results = run_fewshot_ablation(experiment, models, datasets, args.few_shot_counts)
    else:
        results = experiment.run_all_tasks(models, datasets, save_results=True)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1 SUMMARY")
    logger.info("=" * 60)

    for model, model_results in results.items():
        summary = experiment.get_summary_stats(model_results)
        logger.info(f"\n{model}:")
        logger.info(f"  Tasks: {summary['num_tasks']}")
        logger.info(f"  Mean Accuracy: {summary['mean_accuracy']:.2%}")
        logger.info(f"  Min/Max: {summary['min_accuracy']:.2%} / {summary['max_accuracy']:.2%}")
        logger.info(f"  Tasks ≥90%: {summary['num_passing']}/{summary['num_tasks']} ({summary['pass_rate']:.2%})")

    logger.info("\nStep 1 experiments complete!")
    logger.info("Results saved to: results/")


if __name__ == "__main__":
    main()
