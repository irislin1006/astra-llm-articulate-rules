"""Lexical classification tasks."""

from typing import List
from .base import ClassificationTask


class AllLowercaseTask(ClassificationTask):
    """Task: Text is all lowercase."""

    def __init__(self, seed: int = 42):
        super().__init__("all_lowercase", seed)

    def get_rule_description(self) -> str:
        return "The input is labeled as 'True' if and only if the input is all lowercase (contains no uppercase letters)."

    def classify(self, text: str) -> bool:
        return text.islower()

    def get_distractor_rules(self) -> List[str]:
        return [
            "The input is labeled as 'True' if and only if the input has more than 5 words.",
            "The input is labeled as 'True' if and only if the input contains the letter 'e'.",
            "The input is labeled as 'True' if and only if the input ends with a period."
        ]


class AllUppercaseTask(ClassificationTask):
    """Task: Text is all uppercase."""

    def __init__(self, seed: int = 42):
        super().__init__("all_uppercase", seed)

    def get_rule_description(self) -> str:
        return "The input is labeled as 'True' if and only if the input is all uppercase (contains no lowercase letters)."

    def classify(self, text: str) -> bool:
        return text.isupper()

    def get_distractor_rules(self) -> List[str]:
        return [
            "The input is labeled as 'True' if and only if the input has fewer than 10 words.",
            "The input is labeled as 'True' if and only if the input contains the letter 'A'.",
            "The input is labeled as 'True' if and only if the input has no punctuation."
        ]


class ContainsExclamationTask(ClassificationTask):
    """Task: Text contains exclamation mark."""

    def __init__(self, seed: int = 42):
        super().__init__("contains_exclamation", seed)

    def get_rule_description(self) -> str:
        return "The input is labeled as 'True' if and only if the input contains an exclamation mark (!)."

    def classify(self, text: str) -> bool:
        return "!" in text

    def get_distractor_rules(self) -> List[str]:
        return [
            "The input is labeled as 'True' if and only if the input ends with punctuation.",
            "The input is labeled as 'True' if and only if the input is all uppercase.",
            "The input is labeled as 'True' if and only if the input has more than 5 words."
        ]


class StartsWithVowelTask(ClassificationTask):
    """Task: Text starts with a vowel."""

    def __init__(self, seed: int = 42):
        super().__init__("starts_with_vowel", seed)
        self.vowels = "aeiouAEIOU"

    def get_rule_description(self) -> str:
        return "The input is labeled as 'True' if and only if the input starts with a vowel (a, e, i, o, or u)."

    def classify(self, text: str) -> bool:
        return len(text) > 0 and text[0] in self.vowels

    def get_distractor_rules(self) -> List[str]:
        return [
            "The input is labeled as 'True' if and only if the input contains a vowel.",
            "The input is labeled as 'True' if and only if the input has more vowels than consonants.",
            "The input is labeled as 'True' if and only if the first word has more than 3 letters."
        ]


class EndsWithVowelTask(ClassificationTask):
    """Task: Text ends with a vowel."""

    def __init__(self, seed: int = 42):
        super().__init__("ends_with_vowel", seed)
        self.vowels = "aeiouAEIOU"

    def get_rule_description(self) -> str:
        return "The input is labeled as 'True' if and only if the last letter of the input is a vowel (a, e, i, o, or u)."

    def classify(self, text: str) -> bool:
        # Remove trailing punctuation and spaces
        cleaned = text.rstrip(".!? ")
        return len(cleaned) > 0 and cleaned[-1] in self.vowels

    def get_distractor_rules(self) -> List[str]:
        return [
            "The input is labeled as 'True' if and only if the input ends with a consonant.",
            "The input is labeled as 'True' if and only if the last word is longer than 4 letters.",
            "The input is labeled as 'True' if and only if the input contains more vowels than consonants."
        ]


class NoRepeatedLettersTask(ClassificationTask):
    """Task: Text has no repeated consecutive letters."""

    def __init__(self, seed: int = 42):
        super().__init__("no_repeated_letters", seed)

    def get_rule_description(self) -> str:
        return "The input is labeled as 'True' if and only if the input has no repeated consecutive letters (e.g., 'book' would be False because of 'oo')."

    def classify(self, text: str) -> bool:
        # Check for repeated consecutive letters
        text_lower = text.lower()
        for i in range(len(text_lower) - 1):
            if text_lower[i].isalpha() and text_lower[i] == text_lower[i + 1]:
                return False
        return True

    def get_distractor_rules(self) -> List[str]:
        return [
            "The input is labeled as 'True' if and only if all words are unique.",
            "The input is labeled as 'True' if and only if the input has no repeated words.",
            "The input is labeled as 'True' if and only if the input has fewer than 6 words."
        ]
