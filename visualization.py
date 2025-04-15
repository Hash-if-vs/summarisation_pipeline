import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from config import config
import pandas as pd
import os


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

    @staticmethod
    def display_examples(
        dialogues: List[str],
        references: List[str],
        predictions: List[str],
        data_type: str,
        sample_size: int,
        num_examples: int = 3,
    ):
        """
        Display input-output examples in a formatted way
        """
        examples_data = []
        for i in range(min(sample_size, len(dialogues))):
            if i <= num_examples:
                print(f"\nExample {i+1}:")
                print("=" * 80)
                print("[Dialogue]:")
                print(dialogues[i])
                print("\n[Reference Summary]:")
                print(references[i])
                print("\n[Generated Summary]:")
                print(predictions[i])
                print("=" * 80)
            examples_data.append(
                {
                    "model": config.MODEL_NAME,
                    "dialogue": dialogues[i],
                    "reference_summary": references[i],
                    "generated_summary": predictions[i],
                    "data_type": data_type,
                }
            )

        df = pd.DataFrame(examples_data)

        output_file = f"results/qualitative_analysis.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if os.path.exists(output_file):
            df.to_csv(output_file, mode="a", header=False, index=False)
        else:
            df.to_csv(output_file, index=False)

    def plot_model_comparison(
        self,
        final_scores: List[Dict[str, Dict[str, float]]],
        model_names: List[str],
        save_path: str = None,
    ):
        """
        Plot comparison of ROUGE scores across different models.

        Args:
            final_scores: List of score dictionaries for each model
            model_names: List of model names corresponding to the scores
            save_path: Optional path to save the figure
        """
        if not final_scores or not model_names:
            raise ValueError("No scores or model names provided for visualization")
        if len(final_scores) != len(model_names):
            raise ValueError("Number of scores must match number of model names")

        metrics = ["rouge1", "rouge2", "rougeL"]
        score_types = ["precision", "recall", "fmeasure"]

        fig, axes = plt.subplots(
            len(metrics), len(score_types), figsize=(24, 16), sharey=True
        )
        fig.suptitle("Model Comparison by ROUGE Metrics", fontsize=18, y=1.03)

        bar_width = 0.5 / len(model_names)
        x = np.arange(len(model_names))

        # Assign distinct colors per model
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(len(model_names))]

        for i, metric in enumerate(metrics):
            for j, score_type in enumerate(score_types):
                ax = axes[i, j]

                # Draw bars for each model
                for k, (model_name, color) in enumerate(zip(model_names, colors)):
                    value = final_scores[k][metric][score_type]
                    bar = ax.bar(
                        j + k * bar_width,  # Use score_type index + offset
                        value,
                        width=bar_width,
                        label=(
                            model_name if i == 0 and j == 0 else ""
                        ),  # Only label once
                        color=color,
                        alpha=0.8,
                    )

                    # Value label
                    ax.text(
                        j + k * bar_width,
                        value + 0.01,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=90,
                    )

                ax.set_title(f"{metric.upper()} - {score_type.capitalize()}")
                ax.set_ylim(0, 1)
                ax.grid(True, axis="y", linestyle="--", alpha=0.5)
                ax.set_xticks([])  # Remove x-axis labels to clean up

        # Add legend below all plots
        handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
        fig.legend(
            handles,
            model_names,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            ncol=1,
            fontsize=12,
            borderaxespad=0.1,
            frameon=True,
        )

        plt.tight_layout(rect=[0, 0, 0.85, 0.95])
        if save_path:
            os.makedirs(
                os.path.dirname(save_path), exist_ok=True
            )  # Ensure directory exists
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            self.logger.info(f"Saved model comparison plot to {save_path}")

        plt.close()

    def plot_clean_vs_unclean_comparison(
        self, results_csv_path: str, save_path: Optional[str] = None
    ):
        """
        Plot a comparison of model performance on clean vs unclean data using subplots per model.

        Args:
            results_csv_path: Path to CSV containing saved evaluation scores
            save_path: Optional path to save the generated plot
        """
        df = pd.read_csv(results_csv_path)

        # Get list of unique models
        model_names = df["Model_Name"].unique()
        metrics = ["ROUGE1", "ROUGE2", "ROUGEL"]
        score_types = ["P", "R", "F"]

        num_models = len(model_names)
        fig, axes = plt.subplots(
            num_models, len(metrics), figsize=(6 * len(metrics), 4 * num_models)
        )

        if num_models == 1:
            axes = np.expand_dims(axes, axis=0)  # ensure 2D indexing

        for row_idx, model in enumerate(model_names):
            model_data = df[df["Model_Name"] == model]
            for col_idx, metric in enumerate(metrics):
                ax = axes[row_idx][col_idx]
                for score_type in score_types:
                    for data_type in ["Clean", "Unclean"]:
                        value = model_data[model_data["Data_Type"] == data_type][
                            f"{metric}_{score_type}"
                        ].values
                        if len(value) > 0:
                            ax.bar(
                                f"{data_type}_{score_type}",
                                value[0],
                                label=f"{data_type} {score_type}",
                                alpha=0.7,
                            )

                ax.set_title(f"{model} - {metric}")
                ax.set_ylim(0, 1)
                ax.grid(True, axis="y", linestyle="--", alpha=0.5)
                if col_idx == 0:
                    ax.set_ylabel("Score")

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=6)
        fig.suptitle("Model Performance on Clean vs Unclean Data", fontsize=16)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            self.logger.info(f"Saved clean vs unclean comparison plot to {save_path}")

        plt.close()


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
            os.makedirs(
                os.path.dirname(save_path), exist_ok=True
            )  # Ensure directory exists
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
            os.makedirs(
                os.path.dirname(save_path), exist_ok=True
            )  # Ensure directory exists
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            self.logger.info(f"Saved statistics plot to {save_path}")
        plt.close()
