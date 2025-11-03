"""Step 1: Test LLM's ability to learn classification tasks in-context."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
import time
from datetime import datetime

from ..llm_clients import get_client_from_config
from ..utils import PromptBuilder

logger = logging.getLogger(__name__)


class ClassificationExperiment:
    """Runs Step 1: In-context classification experiments."""

    def __init__(self, config, results_dir: str = "results"):
        """
        Initialize experiment.

        Args:
            config: Configuration object
            results_dir: Directory to save results
        """
        # breakpoint()
        self.config = config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)

        self.step1_config = config.get_step1_config()
        self.prompt_builder = PromptBuilder()

    def run_single_task(
        self,
        model_name: str,
        dataset: Dict[str, Any],
        rate_limit_delay: float = 0.5
    ) -> Dict[str, Any]:
        """
        Run classification experiment on a single task.

        Args:
            model_name: Model identifier
            dataset: Dataset dict with train/test splits
            rate_limit_delay: Delay between API calls

        Returns:
            Results dict
        """
        task_id = dataset["task_id"]
        logger.info(f"Running classification for {task_id} with {model_name}")

        # Get LLM client
        client = get_client_from_config(model_name, self.config)

        # Sample few-shot examples from training set
        num_examples = self.step1_config["num_train_examples"]
        train_examples = [(ex["text"], ex["label"]) for ex in dataset["train"]]
        few_shot_examples = train_examples[:num_examples]

        # Test on held-out examples
        test_examples = dataset["test"][:self.step1_config["num_test_examples"]]

        predictions = []
        correct = 0

        for test_ex in tqdm(test_examples, desc=f"{task_id[:20]}", leave=False):
            test_input = test_ex["text"]
            true_label = test_ex["label"]

            # Build prompt
            prompt = self.prompt_builder.build_classification_prompt(
                few_shot_examples,
                test_input
            )

            # Get prediction
            try:
                response = client.generate_with_retry(
                    prompt,
                    temperature=self.step1_config["temperature"],
                    max_tokens=self.step1_config["max_tokens"]
                )
                
                predicted_text = response["response"].strip()

                # Parse prediction
                predicted_label = self._parse_classification(predicted_text)

                is_correct = predicted_label == true_label
                if is_correct:
                    correct += 1

                predictions.append({
                    "input": test_input,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "predicted_text": predicted_text,
                    "correct": is_correct,
                    "raw_prompt": prompt,  # Store the full prompt sent to LLM
                    "raw_response": response  # Store complete response including tokens/cost
                })

                # Rate limiting
                time.sleep(rate_limit_delay)

            except Exception as e:
                logger.error(f"Error on example {test_input[:50]}: {e}")
                predictions.append({
                    "input": test_input,
                    "true_label": true_label,
                    "predicted_label": None,
                    "predicted_text": None,
                    "correct": False,
                    "error": str(e)
                })

        accuracy = correct / len(test_examples) if test_examples else 0

        results = {
            "task_id": task_id,
            "model": model_name,
            "accuracy": accuracy,
            "correct": correct,
            "total": len(test_examples),
            "predictions": predictions,
            "usage": client.get_usage_stats(),
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"{task_id} - {model_name}: {accuracy:.2%} accuracy")

        return results

    def _parse_classification(self, response: str) -> bool:
        """
        Parse model response to extract boolean classification.

        Args:
            response: Model response text

        Returns:
            Boolean classification
        """
        response_lower = response.lower().strip()

        # Check for explicit True/False
        if response_lower.startswith("true"):
            return True
        elif response_lower.startswith("false"):
            return False

        # Check for common variations
        if "true" in response_lower and "false" not in response_lower:
            return True
        elif "false" in response_lower and "true" not in response_lower:
            return False

        # Default to False if ambiguous (could also raise error)
        logger.warning(f"Ambiguous classification response: {response}")
        return False

    def run_all_tasks(
        self,
        models: List[str],
        datasets: Dict[str, Dict[str, Any]],
        save_results: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run classification experiments on all tasks and models.

        Args:
            models: List of model names
            datasets: Dict mapping task_id to dataset
            save_results: Whether to save results to disk

        Returns:
            Dict mapping model to list of task results
        """
        all_results = {}
        # breakpoint()
        for model in models:
            logger.info(f"Starting experiments with {model}")
            model_results = []

            for task_id, dataset in datasets.items():
                try:
                    result = self.run_single_task(model, dataset)
                    model_results.append(result)
                except Exception as e:
                    logger.error(f"Failed on {task_id} with {model}: {e}")
                    model_results.append({
                        "task_id": task_id,
                        "model": model,
                        "error": str(e)
                    })

            all_results[model] = model_results

            if save_results:
                self._save_results(model, model_results)

        return all_results

    def _save_results(self, model: str, results: List[Dict[str, Any]]):
        """Save results to disk."""
        # Sanitize model name for filename
        safe_model_name = model.replace("/", "_").replace(":", "_")
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.results_dir / f"step1_{safe_model_name}_{timestamp}.json"

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved Step 1 results to {filepath}")

    def get_summary_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics from results.

        Args:
            results: List of task results

        Returns:
            Summary dict
        """
        accuracies = [r["accuracy"] for r in results if "accuracy" in r]
        passing_tasks = [r for r in results if r.get("accuracy", 0) >= self.step1_config["accuracy_threshold"]]

        return {
            "num_tasks": len(results),
            "mean_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
            "min_accuracy": min(accuracies) if accuracies else 0,
            "max_accuracy": max(accuracies) if accuracies else 0,
            "num_passing": len(passing_tasks),
            "pass_rate": len(passing_tasks) / len(results) if results else 0,
            "threshold": self.step1_config["accuracy_threshold"]
        }
