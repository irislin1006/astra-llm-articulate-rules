"""LLM-based data generation using vLLM and Qwen3-8B."""

import logging
from typing import List, Tuple
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


class LLMDataGenerator:
    """Generates classification task data using local LLM via vLLM."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 2048,
        seed: int = 42
    ):
        """
        Initialize vLLM-based data generator.

        Args:
            model_name: HuggingFace model identifier
            tensor_parallel_size: Number of GPUs to use
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length (reduce if OOM)
            seed: Random seed for reproducibility
        """
        logger.info(f"Initializing vLLM with {model_name}")
        logger.info(f"  max_model_len: {max_model_len}")
        logger.info(f"  gpu_memory_utilization: {gpu_memory_utilization}")

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            seed=seed
        )

        self.sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=512
            # Note: seed removed to allow diverse generations across train/test splits
            # Model initialization seed still ensures reproducible model loading
        )

        logger.info("vLLM initialized successfully")

    def generate_examples(
        self,
        rule_description: str,
        label: bool,
        n: int = 50,
        max_retries: int = 3
    ) -> List[str]:
        """
        Generate examples that match or don't match a classification rule.

        Args:
            rule_description: Natural language description of the rule
            label: True to generate positive examples, False for negative
            n: Number of examples to generate
            max_retries: Maximum retries for generation

        Returns:
            List of generated text examples
        """
        logger.info(f"Generating {n} {'positive' if label else 'negative'} examples for: {rule_description[:50]}...")

        # Build prompt
        if label:
            prompt = self._build_positive_prompt(rule_description, n)
        else:
            prompt = self._build_negative_prompt(rule_description, n)

        # Generate
        for attempt in range(max_retries):
            try:
                outputs = self.llm.generate([prompt], self.sampling_params)
                response = outputs[0].outputs[0].text.strip()

                # Parse examples from response
                examples = self._parse_examples(response)

                if len(examples) >= n:
                    return examples[:n]
                else:
                    logger.warning(f"Only got {len(examples)}/{n} examples on attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        return examples  # Return what we have

            except Exception as e:
                logger.error(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise

        return []

    def _build_positive_prompt(self, rule_description: str, n: int) -> str:
        """Build prompt for generating positive examples."""
        prompt = f"""You are a creative text generator. Generate {n} diverse and natural English sentences that SATISFY the following rule:

Rule: {rule_description}

Requirements:
- Each sentence should be natural and varied
- Sentences should be different from each other
- Cover diverse topics, lengths, and styles
- Each sentence should clearly satisfy the rule

Generate {n} sentences, one per line:"""

        return prompt

    def _build_negative_prompt(self, rule_description: str, n: int) -> str:
        """Build prompt for generating negative examples."""
        prompt = f"""You are a creative text generator. Generate {n} diverse and natural English sentences that DO NOT SATISFY the following rule:

Rule: {rule_description}

Requirements:
- Each sentence should be natural and varied
- Sentences should be different from each other
- Cover diverse topics, lengths, and styles
- Each sentence should clearly VIOLATE the rule

Generate {n} sentences, one per line:"""

        return prompt

    def _parse_examples(self, response: str) -> List[str]:
        """
        Parse examples from LLM response.

        Args:
            response: Raw LLM output

        Returns:
            List of extracted sentences
        """
        lines = response.strip().split('\n')
        examples = []

        for line in lines:
            line = line.strip()

            # Skip empty lines and common numbering patterns
            if not line:
                continue

            # Remove common prefixes like "1.", "1)", "-", "*"
            import re
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            line = re.sub(r'^[-*•]\s*', '', line)
            line = line.strip('"\'')  # Remove quotes if present

            if line and len(line) > 5:  # Basic sanity check
                examples.append(line)

        return examples

    def generate_verified_examples(
        self,
        task,
        label: bool,
        n: int = 50,
        verification_threshold: float = 0.9,
        min_required: int = None
    ) -> List[str]:
        """
        Generate and verify examples match the expected label.
        Keeps generating until we have at least n verified examples.

        Args:
            task: ClassificationTask instance
            label: Expected label
            n: Number of examples to generate
            verification_threshold: Minimum fraction that must be correct (warning only)
            min_required: Minimum required verified examples (defaults to n)

        Returns:
            List of verified examples (at least n, or min_required if set)
        """
        if min_required is None:
            min_required = n

        verified = []
        max_attempts = 5
        attempt = 0

        while len(verified) < min_required and attempt < max_attempts:
            attempt += 1

            # Calculate how many more we need
            remaining_needed = min_required - len(verified)

            # Generate more than needed to account for verification failures
            # Increase multiplier on later attempts
            multiplier = 1.5 + (attempt - 1) * 0.5
            generate_n = max(remaining_needed, int(remaining_needed * multiplier))

            logger.info(f"Attempt {attempt}: Generating {generate_n} examples (need {remaining_needed} more verified)")

            examples = self.generate_examples(
                task.get_rule_description(),
                label,
                n=generate_n
            )

            # Verify each example
            newly_verified = []
            for example in examples:
                try:
                    predicted_label = task.classify(example)
                    if predicted_label == label:
                        newly_verified.append(example)
                except Exception as e:
                    logger.warning(f"Verification failed for '{example[:50]}...': {e}")

            # Add to verified list
            verified.extend(newly_verified)

            # Check verification rate for this batch
            verification_rate = len(newly_verified) / len(examples) if examples else 0
            logger.info(f"Attempt {attempt}: Verified {len(newly_verified)}/{len(examples)} examples ({verification_rate:.2%})")
            logger.info(f"Total verified so far: {len(verified)}/{min_required}")

            if verification_rate < verification_threshold:
                logger.warning(f"Low verification rate: {verification_rate:.2%} < {verification_threshold:.2%}")

        if len(verified) < min_required:
            logger.error(f"Could not generate {min_required} verified examples after {max_attempts} attempts. Got {len(verified)}.")
        else:
            logger.info(f"Successfully generated {len(verified)} verified examples (target: {min_required})")

        return verified[:n] if len(verified) >= n else verified  # Return requested number, or all if less

    def generate_balanced_dataset(
        self,
        task,
        n_per_label: int = 50,
        min_required_per_label: int = None
    ) -> List[Tuple[str, bool]]:
        """
        Generate balanced dataset with equal positive and negative examples.

        Args:
            task: ClassificationTask instance
            n_per_label: Number of examples per label
            min_required_per_label: Minimum required verified examples per label (defaults to n_per_label)

        Returns:
            List of (text, label) tuples
        """
        logger.info(f"Generating balanced dataset for {task.task_id}")

        # Ensure minimum of 50 per label if not specified
        if min_required_per_label is None:
            min_required_per_label = max(n_per_label, 50)

        logger.info(f"Target: {n_per_label} per label, Minimum required: {min_required_per_label} per label")

        positive_examples = self.generate_verified_examples(
            task,
            label=True,
            n=n_per_label,
            min_required=min_required_per_label
        )
        negative_examples = self.generate_verified_examples(
            task,
            label=False,
            n=n_per_label,
            min_required=min_required_per_label
        )

        # Combine and shuffle
        dataset = [(text, True) for text in positive_examples] + \
                  [(text, False) for text in negative_examples]

        import random
        random.shuffle(dataset)

        logger.info(f"Generated {len(dataset)} total examples ({len(positive_examples)} positive, {len(negative_examples)} negative)")

        # Warn if we didn't meet the minimum
        if len(positive_examples) < min_required_per_label or len(negative_examples) < min_required_per_label:
            logger.warning(f"Did not meet minimum requirement of {min_required_per_label} per label!")
            logger.warning(f"Got: {len(positive_examples)} positive, {len(negative_examples)} negative")

        return dataset
