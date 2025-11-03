"""Numerical classification tasks."""

from typing import List
import re
from .base import ClassificationTask


class ContainsNumberTask(ClassificationTask):
    """Task: Text contains a number."""

    def __init__(self, seed: int = 42):
        super().__init__("contains_number", seed)

    def get_rule_description(self) -> str:
        return "The input is labeled as 'True' if and only if the input contains a number (digit)."

    def classify(self, text: str) -> bool:
        return any(char.isdigit() for char in text)

    def get_distractor_rules(self) -> List[str]:
        return [
            "The input is labeled as 'True' if and only if the input contains more than 5 words.",
            "The input is labeled as 'True' if and only if the input has an even number of characters.",
            "The input is labeled as 'True' if and only if the input starts with a capital letter."
        ]


class EvenDigitSumTask(ClassificationTask):
    """Task: Sum of all digits in text is even."""

    def __init__(self, seed: int = 42):
        super().__init__("even_digit_sum", seed)

    def get_rule_description(self) -> str:
        return "The input is labeled as 'True' if and only if the sum of all digits in the input is an even number."

    def classify(self, text: str) -> bool:
        digits = [int(char) for char in text if char.isdigit()]
        if not digits:
            return True  # Sum of zero digits is 0 (even)
        return sum(digits) % 2 == 0

    def get_distractor_rules(self) -> List[str]:
        return [
            "The input is labeled as 'True' if and only if the input contains a number.",
            "The input is labeled as 'True' if and only if all numbers in the input are even.",
            "The input is labeled as 'True' if and only if the input contains at least two numbers."
        ]


class ContainsPrimeTask(ClassificationTask):
    """Task: Text contains a prime number."""

    def __init__(self, seed: int = 42):
        super().__init__("contains_prime", seed)
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

    def get_rule_description(self) -> str:
        return "The input is labeled as 'True' if and only if the input contains a prime number."

    def _is_prime(self, n: int) -> bool:
        """Check if a number is prime."""
        if n < 2:
            return False
        if n in self.primes:
            return True
        for p in self.primes:
            if p * p > n:
                break
            if n % p == 0:
                return False
        return True

    def classify(self, text: str) -> bool:
        # Extract all numbers
        numbers = re.findall(r'\d+', text)
        for num_str in numbers:
            if self._is_prime(int(num_str)):
                return True
        return False

    def get_distractor_rules(self) -> List[str]:
        return [
            "The input is labeled as 'True' if and only if the input contains an odd number.",
            "The input is labeled as 'True' if and only if the input contains a number greater than 5.",
            "The input is labeled as 'True' if and only if the input contains a single-digit number."
        ]


class EvenWordCountTask(ClassificationTask):
    """Task: Text has an even number of words."""

    def __init__(self, seed: int = 42):
        super().__init__("even_word_count", seed)

    def get_rule_description(self) -> str:
        return "The input is labeled as 'True' if and only if the input has an even number of words."

    def classify(self, text: str) -> bool:
        words = text.split()
        return len(words) % 2 == 0

    def get_distractor_rules(self) -> List[str]:
        return [
            "The input is labeled as 'True' if and only if the input has more than 5 words.",
            "The input is labeled as 'True' if and only if the input has an even number of characters.",
            "The input is labeled as 'True' if and only if the first word has an even number of letters."
        ]
