"""Run Step 3: Faithfulness experiments."""

import logging
import argparse
from pathlib import Path
from src.experiments import FaithfulnessExperiment
from src.utils import get_config, load_all_datasets

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run Step 3 experiments."""
    parser = argparse.ArgumentParser(description="Run Step 3: Faithfulness experiments")
    parser.add_argument(
        "--model",
        required=True,
        help="Model to test (must have Step 2 results)"
    )
    parser.add_argument(
        "--step2-mode",
        choices=["multiple_choice", "free_form"],
        default="multiple_choice",
        help="Which Step 2 results to use"
    )
    args = parser.parse_args()

    logger.info("Starting Step 3: Faithfulness experiments")

    # Load configuration
    config = get_config()

    model = args.model
    logger.info(f"Testing model: {model}")

    # Find Step 2 results
    safe_model_name = model.replace("/", "_").replace(":", "_")
    step2_path = Path("results") / f"step2_{args.step2_mode}_{safe_model_name}.json"

    if not step2_path.exists():
        logger.error(f"Step 2 results not found: {step2_path}")
        logger.error("Run run_step2.py first!")
        return

    logger.info(f"Using Step 2 results from: {step2_path}")

    # Load datasets
    datasets = load_all_datasets("data")
    if not datasets:
        logger.error("No datasets found! Run generate_data.py first.")
        return

    logger.info(f"Loaded {len(datasets)} datasets")

    # Initialize experiment
    experiment = FaithfulnessExperiment(config)

    # Run experiments
    results = experiment.run_from_step2_results(
        model,
        str(step2_path),
        datasets,
        save_results=True
    )

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3 SUMMARY")
    logger.info("=" * 60)

    summary = experiment.get_summary_stats(results)

    logger.info(f"\n{model}:")
    logger.info(f"  Tasks Tested: {summary['num_tasks']}")
    logger.info(f"  Faithful: {summary['num_faithful']}/{summary['num_tasks']} ({summary['faithfulness_rate']:.2%})")
    logger.info(f"  Can Understand Correct Rule: {summary['num_understand_correct']}/{summary['num_tasks']} ({summary['understand_rate']:.2%})")
    logger.info(f"  Mean Articulation Accuracy: {summary['mean_articulation_accuracy']:.2%}")
    logger.info(f"  Mean Alternative Context Accuracy: {summary['mean_alternative_accuracy']:.2%}")

    logger.info("\nStep 3 experiments complete!")
    logger.info("Results saved to: results/")


if __name__ == "__main__":
    main()
