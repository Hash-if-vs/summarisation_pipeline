from rouge_score import rouge_scorer
import numpy as np
import logging
from typing import List, Dict
from config import config


class SummarizationEvaluator:
    """
    Handles evaluation of summarization quality using ROUGE metrics.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(config.LOG_LEVEL)
        self.scorer = rouge_scorer.RougeScorer(config.METRICS, use_stemmer=True)

    def evaluate_batch(
        self, references: List[str], predictions: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a batch of reference and predicted summaries.

        Args:
            references: List of reference (gold) summaries
            predictions: List of predicted (generated) summaries

        Returns:
            Dictionary of ROUGE scores for each metric
        """
        self.logger.info("Evaluating %d summary pairs", len(references))

        if len(references) != len(predictions):
            raise ValueError("References and predictions must be of the same length")

        aggregated_scores = {
            metric: {"precision": [], "recall": [], "fmeasure": []}
            for metric in config.METRICS
        }

        for ref, pred in zip(references, predictions):
            scores = self.scorer.score(ref, pred)

            for metric in config.METRICS:
                aggregated_scores[metric]["precision"].append(scores[metric].precision)
                aggregated_scores[metric]["recall"].append(scores[metric].recall)
                aggregated_scores[metric]["fmeasure"].append(scores[metric].fmeasure)

        # Calculate averages
        avg_scores = {}
        for metric in config.METRICS:
            avg_scores[metric] = {
                "precision": np.mean(aggregated_scores[metric]["precision"]),
                "recall": np.mean(aggregated_scores[metric]["recall"]),
                "fmeasure": np.mean(aggregated_scores[metric]["fmeasure"]),
            }

        return avg_scores

    def print_evaluation(self, scores: Dict[str, Dict[str, float]]):
        """
        Print evaluation results in a readable format.
        """
        print("\nEvaluation Results:")
        print("=" * 50)
        for metric, values in scores.items():
            print(
                f"{metric.upper():<10} Precision: {values['precision']:.4f} \t"
                f"Recall: {values['recall']:.4f} \t"
                f"F1: {values['fmeasure']:.4f}"
            )
        print("=" * 50)


if __name__ == "__main__":
    evaluator = SummarizationEvaluator()
    # Example usage
    input_references = ["This is a reference summary."]
    generated_predictions = ["This is a predicted summary."]
    generated_scores = evaluator.evaluate_batch(input_references, generated_predictions)
    evaluator.print_evaluation(generated_scores)
