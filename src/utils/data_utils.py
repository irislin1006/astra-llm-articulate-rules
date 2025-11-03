"""Data generation and management utilities."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Generates and manages datasets for classification tasks."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize dataset generator.

        Args:
            data_dir: Directory to save datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)

    def generate_dataset(
        self,
        task,
        train_size: int = 100,
        test_size: int = 100,
        save: bool = True
    ) -> Dict[str, Any]:
        """
        Generate train/test split for a task.

        Args:
            task: ClassificationTask instance
            train_size: Number of training examples
            test_size: Number of test examples
            save: Whether to save to disk

        Returns:
            Dict with train and test data
        """
        logger.info(f"Generating dataset for {task.task_id}: {train_size} train, {test_size} test")

        # Generate labeled examples
        train_examples = task.generate_labeled_examples(train_size)
        test_examples = task.generate_labeled_examples(test_size)

        dataset = {
            "task_id": task.task_id,
            "rule": task.get_rule_description(),
            "seed": task.seed,
            "train": [{"text": text, "label": label} for text, label in train_examples],
            "test": [{"text": text, "label": label} for text, label in test_examples],
            "distractors": task.get_distractor_rules()
        }

        if save:
            self._save_dataset(dataset, task.task_id)

        return dataset

    def _save_dataset(self, dataset: Dict[str, Any], task_id: str):
        """Save dataset to disk."""
        filepath = self.data_dir / f"{task_id}.json"
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        logger.info(f"Saved dataset to {filepath}")

    def load_dataset(self, task_id: str) -> Dict[str, Any]:
        """
        Load dataset from disk.

        Args:
            task_id: Task identifier

        Returns:
            Dataset dict
        """
        filepath = self.data_dir / f"{task_id}.json"
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")

        with open(filepath, 'r') as f:
            return json.load(f)

    def dataset_exists(self, task_id: str) -> bool:
        """Check if dataset exists on disk."""
        filepath = self.data_dir / f"{task_id}.json"
        return filepath.exists()

    def generate_all_datasets(self, tasks: List, train_size: int = 100, test_size: int = 100):
        """
        Generate datasets for multiple tasks.

        Args:
            tasks: List of ClassificationTask instances
            train_size: Number of training examples per task
            test_size: Number of test examples per task

        Returns:
            Dict mapping task_id to dataset
        """
        datasets = {}
        for task in tasks:
            datasets[task.task_id] = self.generate_dataset(task, train_size, test_size)

        logger.info(f"Generated {len(datasets)} datasets")
        return datasets


def format_examples_for_prompt(examples: List[Tuple[str, bool]], include_labels: bool = True) -> str:
    """
    Format examples for inclusion in prompts.

    Args:
        examples: List of (text, label) tuples
        include_labels: Whether to include labels

    Returns:
        Formatted string
    """
    lines = []
    for text, label in examples:
        if include_labels:
            lines.append(f'Input: "{text}" Label: {label}')
        else:
            lines.append(f'Input: "{text}"')

    return "\n".join(lines)


def load_all_datasets(data_dir: str = "data") -> Dict[str, Dict[str, Any]]:
    """
    Load all datasets from directory.

    Args:
        data_dir: Directory containing datasets

    Returns:
        Dict mapping task_id to dataset
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return {}

    datasets = {}
    for filepath in data_path.glob("*.json"):
        with open(filepath, 'r') as f:
            dataset = json.load(f)
            datasets[dataset["task_id"]] = dataset

    logger.info(f"Loaded {len(datasets)} datasets from {data_dir}")
    return datasets
