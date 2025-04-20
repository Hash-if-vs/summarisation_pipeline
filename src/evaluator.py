"""
Module for evaluating text summarization quality using ROUGE metrics.
Provides functionality to compare predicted summaries against reference summaries.
"""

import logging
from typing import List, Dict

import numpy as np
from rouge_score import rouge_scorer

from config import config


class SummarizationEvaluator:
    """
    Handles evaluation of summarization quality using ROUGE metrics.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(config.log_level)
        self.scorer = rouge_scorer.RougeScorer(config.metrics, use_stemmer=True)

    def evaluate_batch(
        self, reference_texts: List[str], prediction_texts: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a batch of reference and predicted summaries.

        Args:
            reference_texts: List of reference (gold) summaries
            prediction_texts: List of predicted (generated) summaries

        Returns:
            Dictionary of ROUGE scores for each metric
        """
        self.logger.info("Evaluating %d summary pairs", len(reference_texts))

        if len(reference_texts) != len(prediction_texts):
            raise ValueError("References and predictions must be of the same length")

        aggregated_scores = {
            metric: {"precision": [], "recall": [], "fmeasure": []}
            for metric in config.metrics
        }

        for ref, pred in zip(reference_texts, prediction_texts):
            score_results = self.scorer.score(ref, pred)

            for metric in config.metrics:
                aggregated_scores[metric]["precision"].append(
                    score_results[metric].precision
                )
                aggregated_scores[metric]["recall"].append(score_results[metric].recall)
                aggregated_scores[metric]["fmeasure"].append(
                    score_results[metric].fmeasure
                )

        # Calculate averages
        avg_scores = {}
        for metric in config.metrics:
            avg_scores[metric] = {
                "precision": np.mean(aggregated_scores[metric]["precision"]),
                "recall": np.mean(aggregated_scores[metric]["recall"]),
                "fmeasure": np.mean(aggregated_scores[metric]["fmeasure"]),
            }

        return avg_scores

    def print_evaluation(self, evaluation_scores: Dict[str, Dict[str, float]]):
        """
        Print evaluation results in a readable format.
        """
        print("\nEvaluation Results:")
        print("=" * 50)
        for metric, values in evaluation_scores.items():
            print(
                f"{metric.upper():<10} Precision: {values['precision']:.4f} \t"
                f"Recall: {values['recall']:.4f} \t"
                f"F1: {values['fmeasure']:.4f}"
            )
        print("=" * 50)


if __name__ == "__main__":
    evaluator = SummarizationEvaluator()
    # Example usage
    references = ["This is a reference summary."]
    predictions = ["This is a predicted summary."]
    scores = evaluator.evaluate_batch(references, predictions)
    evaluator.print_evaluation(scores)
