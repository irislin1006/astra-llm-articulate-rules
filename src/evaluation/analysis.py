"""Analysis and visualization utilities for experiment results."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyzes experiment results and generates summary statistics."""

    def __init__(self, results_dir: str = "results"):
        """
        Initialize analyzer.

        Args:
            results_dir: Directory containing result files
        """
        self.results_dir = Path(results_dir)

    def load_step1_results(self, model_name: str) -> List[Dict[str, Any]]:
        """Load Step 1 results for a model."""
        safe_name = model_name.replace("/", "_").replace(":", "_")
        filepath = self.results_dir / f"step1_{safe_name}.json"

        if not filepath.exists():
            logger.warning(f"No Step 1 results found for {model_name}")
            return []

        with open(filepath, 'r') as f:
            return json.load(f)

    def load_step2_results(self, model_name: str, mode: str = "free_form") -> List[Dict[str, Any]]:
        """Load Step 2 results for a model."""
        safe_name = model_name.replace("/", "_").replace(":", "_")
        filepath = self.results_dir / f"step2_{mode}_{safe_name}.json"

        if not filepath.exists():
            logger.warning(f"No Step 2 {mode} results found for {model_name}")
            return []

        with open(filepath, 'r') as f:
            return json.load(f)

    def load_step3_results(self, model_name: str) -> List[Dict[str, Any]]:
        """Load Step 3 results for a model."""
        safe_name = model_name.replace("/", "_").replace(":", "_")
        filepath = self.results_dir / f"step3_{safe_name}.json"

        if not filepath.exists():
            logger.warning(f"No Step 3 results found for {model_name}")
            return []

        with open(filepath, 'r') as f:
            return json.load(f)

    def create_step1_summary_table(self, models: List[str]) -> pd.DataFrame:
        """
        Create summary table for Step 1 results across models.

        Args:
            models: List of model names

        Returns:
            DataFrame with summary statistics
        """
        rows = []

        for model in models:
            results = self.load_step1_results(model)

            if not results:
                continue

            accuracies = [r["accuracy"] for r in results if "accuracy" in r]
            passing_90 = sum(1 for acc in accuracies if acc >= 0.9)

            rows.append({
                "Model": model,
                "Tasks": len(results),
                "Mean Accuracy": f"{sum(accuracies) / len(accuracies):.2%}" if accuracies else "N/A",
                "Min Accuracy": f"{min(accuracies):.2%}" if accuracies else "N/A",
                "Max Accuracy": f"{max(accuracies):.2%}" if accuracies else "N/A",
                "Tasks ≥90%": f"{passing_90}/{len(results)}",
                "Pass Rate": f"{passing_90 / len(results):.2%}" if results else "N/A"
            })

        return pd.DataFrame(rows)

    def create_step2_summary_table(self, models: List[str], mode: str = "free_form") -> pd.DataFrame:
        """
        Create summary table for Step 2 results.

        Args:
            models: List of model names
            mode: "multiple_choice" or "free_form"

        Returns:
            DataFrame with summary statistics
        """
        rows = []

        for model in models:
            results = self.load_step2_results(model, mode)

            if not results:
                continue

            if mode == "multiple_choice":
                correct = sum(1 for r in results if r.get("correct", False))
                total = len([r for r in results if "correct" in r])

                rows.append({
                    "Model": model,
                    "Mode": mode,
                    "Tasks": total,
                    "Accuracy": f"{correct / total:.2%}" if total > 0 else "N/A",
                    "Correct": f"{correct}/{total}"
                })

            elif mode == "free_form":
                similarities = [r.get("best_similarity", 0) for r in results if "best_similarity" in r]

                rows.append({
                    "Model": model,
                    "Mode": mode,
                    "Tasks": len(similarities),
                    "Mean Similarity": f"{sum(similarities) / len(similarities):.3f}" if similarities else "N/A",
                    "Median Similarity": f"{sorted(similarities)[len(similarities) // 2]:.3f}" if similarities else "N/A"
                })

        return pd.DataFrame(rows)

    def create_step3_summary_table(self, models: List[str]) -> pd.DataFrame:
        """
        Create summary table for Step 3 results.

        Args:
            models: List of model names

        Returns:
            DataFrame with summary statistics
        """
        rows = []

        for model in models:
            results = self.load_step3_results(model)

            if not results:
                continue

            faithful = sum(1 for r in results if r.get("is_faithful", False))
            understand = sum(1 for r in results if r.get("can_understand_correct_rule", False))

            art_accs = [r["articulation_test"]["accuracy"] for r in results if "articulation_test" in r]
            alt_accs = [r["alternative_context_test"]["accuracy"] for r in results if "alternative_context_test" in r]

            rows.append({
                "Model": model,
                "Tasks": len(results),
                "Faithful": f"{faithful}/{len(results)}",
                "Faithfulness Rate": f"{faithful / len(results):.2%}" if results else "N/A",
                "Can Understand Rule": f"{understand}/{len(results)}",
                "Understand Rate": f"{understand / len(results):.2%}" if results else "N/A",
                "Mean Articulation Acc": f"{sum(art_accs) / len(art_accs):.2%}" if art_accs else "N/A",
                "Mean Alternative Acc": f"{sum(alt_accs) / len(alt_accs):.2%}" if alt_accs else "N/A"
            })

        return pd.DataFrame(rows)

    def create_per_task_comparison(self, models: List[str], step: int = 1) -> pd.DataFrame:
        """
        Create per-task comparison across models.

        Args:
            models: List of model names
            step: Which step (1, 2, or 3)

        Returns:
            DataFrame with per-task results
        """
        if step == 1:
            return self._create_step1_task_comparison(models)
        elif step == 2:
            return self._create_step2_task_comparison(models)
        elif step == 3:
            return self._create_step3_task_comparison(models)
        else:
            raise ValueError(f"Invalid step: {step}")

    def _create_step1_task_comparison(self, models: List[str]) -> pd.DataFrame:
        """Create per-task Step 1 comparison."""
        task_data = {}

        for model in models:
            results = self.load_step1_results(model)
            for result in results:
                task_id = result.get("task_id")
                if task_id:
                    if task_id not in task_data:
                        task_data[task_id] = {}
                    task_data[task_id][model] = result.get("accuracy", 0)

        rows = []
        for task_id, model_results in task_data.items():
            row = {"Task": task_id}
            for model in models:
                acc = model_results.get(model, 0)
                row[model] = f"{acc:.2%}"
            rows.append(row)

        return pd.DataFrame(rows)

    def _create_step2_task_comparison(self, models: List[str]) -> pd.DataFrame:
        """Create per-task Step 2 comparison."""
        # Similar implementation for Step 2
        pass

    def _create_step3_task_comparison(self, models: List[str]) -> pd.DataFrame:
        """Create per-task Step 3 comparison."""
        # Similar implementation for Step 3
        pass

    def export_summary_csv(self, output_dir: str = "results"):
        """Export all summary tables to CSV."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Find all models
        models = set()
        for filepath in self.results_dir.glob("step1_*.json"):
            model_name = filepath.stem.replace("step1_", "").replace("_", "/")
            models.add(model_name)

        models = sorted(list(models))

        if not models:
            logger.warning("No results found")
            return

        # Step 1 summary
        step1_df = self.create_step1_summary_table(models)
        step1_df.to_csv(output_path / "step1_summary.csv", index=False)
        logger.info(f"Saved Step 1 summary to {output_path / 'step1_summary.csv'}")

        # Step 2 summaries
        for mode in ["multiple_choice", "free_form"]:
            step2_df = self.create_step2_summary_table(models, mode)
            if not step2_df.empty:
                step2_df.to_csv(output_path / f"step2_{mode}_summary.csv", index=False)
                logger.info(f"Saved Step 2 {mode} summary")

        # Step 3 summary
        step3_df = self.create_step3_summary_table(models)
        if not step3_df.empty:
            step3_df.to_csv(output_path / "step3_summary.csv", index=False)
            logger.info(f"Saved Step 3 summary")
