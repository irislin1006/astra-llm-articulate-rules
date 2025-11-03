"""Step 2: Test LLM's ability to articulate classification rules."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import time
import random
from datetime import datetime

from ..llm_clients import get_client_from_config
from ..utils import PromptBuilder

logger = logging.getLogger(__name__)


class ArticulationExperiment:
    """Runs Step 2: Rule articulation experiments."""

    def __init__(self, config, results_dir: str = "results"):
        """
        Initialize experiment.

        Args:
            config: Configuration object
            results_dir: Directory to save results
        """
        self.config = config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)

        self.step2_config = config.get_step2_config()
        self.prompt_builder = PromptBuilder()

    def run_multiple_choice(
        self,
        model_name: str,
        dataset: Dict[str, Any],
        rate_limit_delay: float = 0.5
    ) -> Dict[str, Any]:
        """
        Run multiple-choice articulation experiment.

        Args:
            model_name: Model identifier
            dataset: Dataset dict
            rate_limit_delay: Delay between API calls

        Returns:
            Results dict
        """
        task_id = dataset["task_id"]
        logger.info(f"Running multiple-choice articulation for {task_id} with {model_name}")

        # Get LLM client
        client = get_client_from_config(model_name, self.config)

        # Sample few-shot examples
        num_examples = self.step2_config["num_train_examples"]
        train_examples = [(ex["text"], ex["label"]) for ex in dataset["train"]]
        few_shot_examples = train_examples[:num_examples]

        # Get correct rule and distractors
        correct_rule = dataset["rule"]
        distractors = dataset.get("distractors", [])[:self.step2_config["multiple_choice"]["num_distractors"]]

        # Build prompt
        prompt, correct_idx = self.prompt_builder.build_articulation_prompt_multiple_choice(
            few_shot_examples,
            correct_rule,
            distractors,
            shuffle=True
        )
        # breakpoint()
        # Get prediction
        try:
            response = client.generate_with_retry(
                prompt,
                temperature=self.step2_config["temperature"],
                max_tokens=self.step2_config["max_tokens"]
            )

            predicted_text = response["response"].strip()
            # breakpoint()
            predicted_idx = self._parse_multiple_choice(predicted_text, len(distractors) + 1)
            # breakpoint()
            is_correct = predicted_idx == correct_idx

            result = {
                "task_id": task_id,
                "model": model_name,
                "mode": "multiple_choice",
                "correct_rule": correct_rule,
                "correct_idx": correct_idx,
                "predicted_idx": predicted_idx,
                "predicted_text": predicted_text,
                "correct": is_correct,
                "raw_prompt": prompt,  # Store the full prompt sent to LLM
                "raw_response": response,  # Store complete response including tokens/cost
                "usage": client.get_usage_stats(),
                "timestamp": datetime.now().isoformat()
            }
            # breakpoint()
            time.sleep(rate_limit_delay)

        except Exception as e:
            logger.error(f"Error on {task_id}: {e}")
            result = {
                "task_id": task_id,
                "model": model_name,
                "mode": "multiple_choice",
                "error": str(e)
            }

        return result

    def run_free_form(
        self,
        model_name: str,
        dataset: Dict[str, Any],
        variations: List[str] = None,
        rate_limit_delay: float = 0.5
    ) -> Dict[str, Any]:
        """
        Run free-form articulation experiment.

        Args:
            model_name: Model identifier
            dataset: Dataset dict
            variations: Prompt variations to try
            rate_limit_delay: Delay between API calls

        Returns:
            Results dict
        """
        task_id = dataset["task_id"]
        logger.info(f"Running free-form articulation for {task_id} with {model_name}")

        if variations is None:
            variations = ["basic", "detailed", "cot"]

        # Get LLM client
        client = get_client_from_config(model_name, self.config)

        # Sample few-shot examples
        num_examples = self.step2_config["num_train_examples"]
        train_examples = [(ex["text"], ex["label"]) for ex in dataset["train"]]
        few_shot_examples = train_examples[:num_examples]

        correct_rule = dataset["rule"]

        results_by_variation = []

        for variation in variations:
            try:
                # Build prompt
                prompt = self.prompt_builder.build_articulation_prompt_freeform(
                    few_shot_examples,
                    variation=variation
                )

                # Get prediction
                response = client.generate_with_retry(
                    prompt,
                    temperature=self.step2_config["temperature"],
                    max_tokens=self.step2_config["max_tokens"]
                )

                articulated_rule = response["response"].strip()

                # Simple similarity check (can be improved with semantic similarity)
                similarity = self._compute_rule_similarity(correct_rule, articulated_rule)

                results_by_variation.append({
                    "variation": variation,
                    "articulated_rule": articulated_rule,
                    "similarity": similarity,
                    "tokens": response["tokens"],
                    "raw_prompt": prompt,  # Store the full prompt sent to LLM
                    "raw_response": response  # Store complete response including tokens/cost
                })

                time.sleep(rate_limit_delay)

            except Exception as e:
                logger.error(f"Error on {task_id} variation {variation}: {e}")
                results_by_variation.append({
                    "variation": variation,
                    "error": str(e)
                })

        # Find best variation
        best_variation = max(
            [r for r in results_by_variation if "similarity" in r],
            key=lambda x: x["similarity"],
            default=None
        )

        result = {
            "task_id": task_id,
            "model": model_name,
            "mode": "free_form",
            "correct_rule": correct_rule,
            "results_by_variation": results_by_variation,
            "best_variation": best_variation["variation"] if best_variation else None,
            "best_articulation": best_variation["articulated_rule"] if best_variation else None,
            "best_similarity": best_variation["similarity"] if best_variation else 0,
            "usage": client.get_usage_stats(),
            "timestamp": datetime.now().isoformat()
        }

        return result

    def _parse_multiple_choice(self, response: str, num_options: int) -> Optional[int]:
        """
        Parse multiple-choice response.

        Args:
            response: Model response
            num_options: Number of options

        Returns:
            Selected option index (0-based) or None
        """
        print(f"\n\nParsing multiple-choice response: {response}\n\n")
        response_upper = response.upper().strip()

        # Look for letter (A, B, C, D)
        for i in range(num_options):
            letter = chr(65 + i)  # A, B, C, D...
            if response_upper.startswith(letter) or f"({letter})" in response_upper:
                print(f"Found letter: {letter}")
                return i

        logger.warning(f"Could not parse multiple-choice response: {response}")
        return None

    def _compute_rule_similarity(self, rule1: str, rule2: str) -> float:
        """
        Compute similarity between two rule descriptions.

        This is a simple lexical similarity. For better results, could use:
        - Semantic similarity (sentence embeddings)
        - LLM-as-judge
        - Manual annotation

        Args:
            rule1: First rule
            rule2: Second rule

        Returns:
            Similarity score (0-1)
        """
        # Normalize
        r1 = set(rule1.lower().split())
        r2 = set(rule2.lower().split())

        # Jaccard similarity
        if not r1 or not r2:
            return 0.0

        intersection = len(r1 & r2)
        union = len(r1 | r2)

        return intersection / union if union > 0 else 0.0

    def run_all_tasks(
        self,
        models: List[str],
        datasets: Dict[str, Dict[str, Any]],
        mode: str = "both",
        save_results: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run articulation experiments on all tasks.

        Args:
            models: List of model names
            datasets: Dict mapping task_id to dataset
            mode: "multiple_choice", "free_form", or "both"
            save_results: Whether to save results

        Returns:
            Dict mapping model to list of results
        """
        all_results = {}

        for model in models:
            logger.info(f"Starting Step 2 ({mode}) with {model}")
            model_results = []

            for task_id, dataset in tqdm(datasets.items(), desc=model):
                try:
                    if mode in ["multiple_choice", "both"]:
                        mc_result = self.run_multiple_choice(model, dataset)
                        model_results.append(mc_result)

                    if mode in ["free_form", "both"]:
                        ff_result = self.run_free_form(model, dataset)
                        model_results.append(ff_result)

                except Exception as e:
                    logger.error(f"Failed on {task_id} with {model}: {e}")
                    model_results.append({
                        "task_id": task_id,
                        "model": model,
                        "error": str(e)
                    })

            all_results[model] = model_results

            if save_results:
                self._save_results(model, mode, model_results)

        return all_results

    def _save_results(self, model: str, mode: str, results: List[Dict[str, Any]]):
        """Save results to disk."""
        safe_model_name = model.replace("/", "_").replace(":", "_")
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.results_dir / f"step2_{mode}_{safe_model_name}_{timestamp}.json"

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved Step 2 results to {filepath}")

    def get_summary_stats(self, results: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
        """Get summary statistics."""
        if mode == "multiple_choice":
            correct_count = sum(1 for r in results if r.get("correct", False))
            total = len([r for r in results if "correct" in r])
            return {
                "accuracy": correct_count / total if total > 0 else 0,
                "correct": correct_count,
                "total": total
            }
        elif mode == "free_form":
            similarities = [r.get("best_similarity", 0) for r in results if "best_similarity" in r]
            return {
                "mean_similarity": sum(similarities) / len(similarities) if similarities else 0,
                "median_similarity": sorted(similarities)[len(similarities) // 2] if similarities else 0,
                "num_tasks": len(similarities)
            }
        else:
            return {}
