import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


class SummaryVisualizer:
    """
    Handles visualization of evaluation results and examples
    """

    @staticmethod
    def plot_rouge_scores(scores: Dict[str, Dict[str, float]], save_path: str = None):
        """
        Plot ROUGE scores from evaluation results.

        Args:
            scores: Dictionary of ROUGE scores in format:
                   {'rouge1': {'precision': 0.8, 'recall': 0.7, 'fmeasure': 0.75}, ...}
            save_path: Optional path to save the figure
        """
        if not scores:
            raise ValueError("No scores provided for visualization")

        metrics = list(scores.keys())
        x = np.arange(len(metrics))  # the label locations
        width = 0.25  # the width of the bars

        fig, ax = plt.subplots(figsize=(12, 6))

        # Create bars for each metric type
        precision_bars = ax.bar(
            x - width,
            [scores[m]["precision"] for m in metrics],
            width,
            label="Precision",
            color="#1f77b4",
        )
        recall_bars = ax.bar(
            x,
            [scores[m]["recall"] for m in metrics],
            width,
            label="Recall",
            color="#ff7f0e",
        )
        f1_bars = ax.bar(
            x + width,
            [scores[m]["fmeasure"] for m in metrics],
            width,
            label="F1",
            color="#2ca02c",
        )

        # Add labels and title
        ax.set_xlabel("ROUGE Metrics")
        ax.set_ylabel("Scores")
        ax.set_title("ROUGE Score Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)

        # Add value labels on top of each bar
        for bars in [precision_bars, recall_bars, f1_bars]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        plt.show()

    @staticmethod
    def display_examples(
        dialogues: List[str],
        references: List[str],
        predictions: List[str],
        num_examples: int = 3,
    ):
        """
        Display input-output examples in a formatted way
        """
        for i in range(min(num_examples, len(dialogues))):
            print(f"\nExample {i+1}:")
            print("=" * 80)
            print("[Dialogue]:")
            print(dialogues[i])
            print("\n[Reference Summary]:")
            print(references[i])
            print("\n[Generated Summary]:")
            print(predictions[i])
            print("=" * 80)
