import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict


class SummaryVisualizer:
    """
    Handles visualization of evaluation results and examples
    """

    @staticmethod
    def plot_rouge_scores(results: List[Dict], save_path: str = None):
        """
        Plot ROUGE scores from tuning results

        Args:
            results: List of tuning results from HyperparameterTuner
            save_path: Optional path to save the figure
        """
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(
            [
                {
                    **r["params"],
                    **{"rouge1": r["scores"]["rouge1"]["fmeasure"]},
                    **{"rouge2": r["scores"]["rouge2"]["fmeasure"]},
                    **{"rougeL": r["scores"]["rougeL"]["fmeasure"]},
                }
                for r in results
            ]
        )

        # Plot settings
        plt.figure(figsize=(12, 6))

        # Plot each metric
        for i, metric in enumerate(["rouge1", "rouge2", "rougeL"], 1):
            plt.subplot(1, 3, i)
            plt.scatter(
                df["max_length"], df["num_beams"], c=df[metric], cmap="viridis", s=100
            )
            plt.colorbar(label=f"{metric} F1")
            plt.xlabel("Max Length")
            plt.ylabel("Num Beams")
            plt.title(f"{metric.upper()} Scores")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
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

        Args:
            dialogues: List of input dialogues
            references: List of reference summaries
            predictions: List of predicted summaries
            num_examples: Number of examples to display
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
