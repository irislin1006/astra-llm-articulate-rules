"""Base class for classification tasks."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import random


class ClassificationTask(ABC):
    """Abstract base class for classification tasks."""

    def __init__(self, task_id: str, seed: int = 42):
        """
        Initialize task.

        Args:
            task_id: Unique identifier for the task
            seed: Random seed for reproducibility
        """
        self.task_id = task_id
        self.seed = seed
        self.rng = random.Random(seed)

    @abstractmethod
    def get_rule_description(self) -> str:
        """
        Get natural language description of the classification rule.

        Returns:
            Human-readable rule description
        """
        pass

    @abstractmethod
    def classify(self, text: str) -> bool:
        """
        Classify a text input.

        Args:
            text: Input text to classify

        Returns:
            True or False classification
        """
        pass

    def get_distractor_rules(self) -> List[str]:
        """
        Get distractor rule descriptions for multiple-choice.

        Returns:
            List of plausible but incorrect rule descriptions
        """
        # Default distractors - tasks can override for better distractors
        return [
            "The input is labeled as 'True' if the input contains the letter 'e'.",
            "The input is labeled as 'True' if the input has an even number of characters.",
            "The input is labeled as 'True' if the input starts with a capital letter.",
            "The input is labeled as 'True' if the input contains a vowel."
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            "task_id": self.task_id,
            "rule": self.get_rule_description(),
            "seed": self.seed
        }

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.task_id})"
