"""Run Step 2: Articulation experiments."""

import logging
import argparse
from src.experiments import ArticulationExperiment
from src.utils import get_config, load_all_datasets

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run Step 2 experiments."""
    parser = argparse.ArgumentParser(description="Run Step 2: Articulation experiments")
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
        "--mode",
        choices=["multiple_choice", "free_form", "both"],
        default="multiple_choice",
        help="Which articulation mode to test"
    )
    args = parser.parse_args()

    logger.info("Starting Step 2: Articulation experiments")

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
    logger.info(f"Mode: {args.mode}")

    # Load datasets
    datasets = load_all_datasets("data")
    if not datasets:
        logger.error("No datasets found! Run generate_data.py first.")
        return

    logger.info(f"Loaded {len(datasets)} datasets")

    # Initialize experiment
    experiment = ArticulationExperiment(config)

    # Run experiments
    results = experiment.run_all_tasks(
        models,
        datasets,
        mode=args.mode,
        save_results=True
    )

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2 SUMMARY")
    logger.info("=" * 60)

    for model, model_results in results.items():
        logger.info(f"\n{model}:")

        # Multiple-choice results
        mc_results = [r for r in model_results if r.get("mode") == "multiple_choice"]
        if mc_results:
            summary = experiment.get_summary_stats(mc_results, "multiple_choice")
            logger.info(f"  Multiple-Choice:")
            logger.info(f"    Accuracy: {summary['accuracy']:.2%}")
            logger.info(f"    Correct: {summary['correct']}/{summary['total']}")

        # Free-form results
        ff_results = [r for r in model_results if r.get("mode") == "free_form"]
        if ff_results:
            summary = experiment.get_summary_stats(ff_results, "free_form")
            logger.info(f"  Free-Form:")
            logger.info(f"    Mean Similarity: {summary['mean_similarity']:.3f}")
            logger.info(f"    Median Similarity: {summary['median_similarity']:.3f}")

    logger.info("\nStep 2 experiments complete!")
    logger.info("Results saved to: results/")


if __name__ == "__main__":
    main()
