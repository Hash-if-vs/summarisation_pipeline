"""Hyper parameter tuning module"""

import itertools
import logging
from typing import Dict, Any
from config import config

from .model import SummarizationModel
from .evaluator import SummarizationEvaluator


class HyperparameterTuner:
    """
    Simple grid search for hyperparameter tuning
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(config.log_level)
        self.model = SummarizationModel()
        self.evaluator = SummarizationEvaluator()

    def tune(
        self, dialogues: list, references: list, param_grid: Dict[str, list]
    ) -> Dict[str, Any]:
        """
        Perform grid search over parameter combinations

        Args:
            dialogues: List of input dialogues
            references: List of reference summaries
            param_grid: Dictionary of parameters to tune with their possible values

        Returns:
            Dictionary with best parameters and corresponding scores
        """
        self.logger.info("Starting hyperparameter tuning")

        # Generate all parameter combinations
        param_names = param_grid.keys()
        param_values = param_grid.values()
        param_combinations = [
            dict(zip(param_names, combo)) for combo in itertools.product(*param_values)
        ]

        best_score = -1
        best_params = {}
        results = []

        # Load model once (outside the loop for efficiency)
        self.model.load_model()

        for params in param_combinations:
            self.logger.info("Testing parameters: %s", params)

            # Update model parameters
            # self.model.update_parameters(params)

            # Generate summaries with current parameters
            summaries = self.model.summarize(dialogues)

            # Evaluate
            scores = self.evaluator.evaluate_batch(references, summaries)
            avg_f1 = sum(score["fmeasure"] for score in scores.values()) / len(scores)

            # Store results
            results.append({"params": params, "scores": scores, "avg_f1": avg_f1})

            # Update best if improved
            if avg_f1 > best_score:
                best_score = avg_f1
                best_params = params

        self.logger.info(
            "Tuning complete. Best F1: %.4f with params: %s", best_score, best_params
        )

        return {
            "best_params": best_params,
            "best_score": best_score,
            "all_results": results,
        }
