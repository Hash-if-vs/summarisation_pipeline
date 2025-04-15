import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from config import config


class SummaryVisualizer:
    """
    Comprehensive visualization handler for:
    - Evaluation results (ROUGE scores)
    - Dataset token distributions
    - Statistical measures comparison
    - Example displays
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(config.LOG_LEVEL)

    def plot_rouge_scores(
        self, scores: Dict[str, Dict[str, float]], save_path: str = None
    ):
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
        plt.close()

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


class TokenDataVisualizer:
    """
    Handles all visualizations related to token length analysis
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.TokenDataVisualizer")
        self.logger.setLevel(config.LOG_LEVEL)
        self.color_palette = {
            "train": {"dialogues": "#1f77b4", "summaries": "#2ca02c"},
            "test": {"dialogues": "#ff7f0e", "summaries": "#d62728"},
        }

    def plot_token_distributions(
        self, stats: Dict[str, Dict[str, Any]], save_path: str = None
    ):
        """
        Plot dialogue and summary length distributions

        Args:
            stats: Dictionary containing token length statistics with raw values
            save_path: Optional path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot dialogue lengths
        for split in ["train", "test"]:
            ax1.hist(
                stats[split]["dialogues"]["values"],
                bins=30,
                alpha=0.6,
                label=f"{split.capitalize()} Set",
                edgecolor="white",
            )
        ax1.set_title("Dialogue Token Lengths")
        ax1.set_xlabel("Number of Tokens")
        ax1.set_ylabel("Frequency")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Plot summary lengths
        for split in ["train", "test"]:
            ax2.hist(
                stats[split]["summaries"]["values"],
                bins=30,
                alpha=0.6,
                label=f"{split.capitalize()} Set",
                edgecolor="white",
            )
        ax2.set_title("Summary Token Lengths")
        ax2.set_xlabel("Number of Tokens")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_statistics(
        self, stats: Dict[str, Dict[str, Any]], save_path: Optional[str] = None
    ) -> None:
        """
        Plot comparison of statistical measures with exact values and proper legend

        Args:
            stats: Dictionary containing token length statistics
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=(20, 10))

        metrics = ["mean", "median", "mode", "max", "min", "std"]
        x = np.arange(len(metrics)) * 2  # Increase spacing between groups
        width = 0.35
        y_offset = 0.05  # Vertical offset for text annotations

        ax = plt.axes([0.1, 0.1, 0.7, 0.8])
        legend_handles = []

        for i, split in enumerate(["train", "test"]):
            bars = ax.bar(
                x - width / 2 + i * width,
                [stats[split]["dialogues"][m] for m in metrics],
                width,
                color=self.color_palette[split]["dialogues"],
                alpha=0.7,
            )

            if i == 0:
                legend_handles.append(bars[0])

            for j, bar in enumerate(bars):
                height = bar.get_height()
                value = stats[split]["dialogues"][metrics[j]]
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + y_offset,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            bars = ax.bar(
                x + width / 2 + i * width,
                [stats[split]["summaries"][m] for m in metrics],
                width,
                color=self.color_palette[split]["summaries"],
                alpha=0.7,
            )

            if i == 0:
                legend_handles.append(bars[0])

            for j, bar in enumerate(bars):
                height = bar.get_height()
                value = stats[split]["summaries"][metrics[j]]
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + y_offset,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.set_title("Token Length Statistics Comparison (Values Shown Above Bars)")
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.set_ylabel("Token Count")
        ax.grid(axis="y", alpha=0.3)

        legend_ax = plt.axes([0.82, 0.1, 0.15, 0.8])
        legend_ax.axis("off")

        legend_labels = [
            "Train Dialogues",
            "Train Summaries",
            "Test Dialogues",
            "Test Summaries",
        ]
        legend_ax.legend(
            legend_handles
            + [
                plt.Rectangle(
                    (0, 0), 1, 1, fc=self.color_palette["test"]["dialogues"], alpha=0.7
                ),
                plt.Rectangle(
                    (0, 0), 1, 1, fc=self.color_palette["test"]["summaries"], alpha=0.7
                ),
            ],
            legend_labels,
            loc="upper left",
            framealpha=0.7,
        )

        # Replacing symbolic representations with full words
        stats_text = ["Statistics Summary:"]
        for split in ["train", "test"]:
            stats_text.append(f"\n{split.capitalize()} Set:")
            stats_text.append(f"Dialogues (Mean / Median / Mode/ Max/ Min/ SD):")
            stats_text.append(
                f"{stats[split]['dialogues']['mean']:.1f} / {stats[split]['dialogues']['median']:.1f} / {stats[split]['dialogues']['mode']}/ {stats[split]['dialogues']['max']}/ {stats[split]['dialogues']['min']}/ {stats[split]['dialogues']['std']:.1f}"
            )
            stats_text.append(f"Summaries (Mean / Median / Mode/ Max/ Min/ SD):")
            stats_text.append(
                f"{stats[split]['summaries']['mean']:.1f} / {stats[split]['summaries']['median']:.1f} / {stats[split]['summaries']['mode']}/ {stats[split]['summaries']['max']}/ {stats[split]['summaries']['min']}/ {stats[split]['summaries']['std']:.1f}"
            )

        legend_ax.text(
            0,
            0.5,
            "\n".join(stats_text),
            ha="left",
            va="center",
            fontfamily="monospace",
            bbox=dict(facecolor="white", alpha=0.7),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            self.logger.info(f"Saved statistics plot to {save_path}")
        plt.close()
