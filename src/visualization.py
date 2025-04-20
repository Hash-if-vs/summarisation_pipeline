"""Visualization utilities for summarization model results and data statistics."""

import logging
import os
from typing import Dict, List, Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.config import config


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
        self.logger.setLevel(config.log_level)

    @staticmethod
    def display_examples(
        sample_data: Tuple[List[str], List[str], List[str]],
        data_type: str,
        sample_size: int,
        num_examples: int = 3,
    ):
        """
        Display input-output examples in a formatted way

        Args:
            sample_data: Tuple containing (dialogues, references, predictions)
            data_type: Type of dataset (e.g. 'clean', 'unclean')
            sample_size: Number of samples to process
            num_examples: Number of examples to print
        """
        dialogues, references, predictions = sample_data
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
                    "model": config.model_name,
                    "dialogue": dialogues[i],
                    "reference_summary": references[i],
                    "generated_summary": predictions[i],
                    "data_type": data_type,
                }
            )

        df = pd.DataFrame(examples_data)
        base_path = config.file_save_paths["generated_summaries"]
        output_file = os.path.join(
            base_path,
            f"{data_type.lower()}_summaries.csv",
        )
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

        # Pre-define metric groups to reduce local variable count
        metrics = ["rouge1", "rouge2", "rougeL"]
        score_types = ["precision", "recall", "fmeasure"]

        # Create distinct colors per model
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(len(model_names))]
        bar_width = 0.5 / len(model_names)

        # Create the figure and axes
        fig, axes = plt.subplots(
            len(metrics), len(score_types), figsize=(24, 16), sharey=True
        )
        fig.suptitle("Model Comparison by ROUGE Metrics", fontsize=18, y=1.03)

        # Plot the data
        self._plot_model_comparison_bars(
            axes, metrics, score_types, final_scores, model_names, colors, bar_width
        )

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
            self.logger.info("Saved model comparison plot to %s", save_path)

        plt.close()

    def _plot_model_comparison_bars(
        self, axes, metrics, score_types, final_scores, model_names, colors, bar_width
    ):
        """
        Helper method to plot the bars for model comparison to reduce complexity

        Args:
            axes: Matplotlib axes to plot on
            metrics: List of ROUGE metrics to display
            score_types: Types of ROUGE scores to display
            final_scores: Model score data
            model_names: Names of models being compared
            colors: Color scheme for models
            bar_width: Width for each bar in the plot
        """
        for i, metric in enumerate(metrics):
            for j, score_type in enumerate(score_types):
                ax = axes[i, j]

                # Draw bars for each model
                for k, (model_name, color) in enumerate(zip(model_names, colors)):
                    value = final_scores[k][metric][score_type]
                    ax.bar(
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

    def plot_clean_vs_unclean_comparison(
        self, results_csv_path: str, save_path: Optional[str] = None
    ):
        """
        Plot a comparison of model performance on clean vs unclean data using subplots per model.

        Args:
            results_csv_path: Path to CSV containing saved evaluation scores
            save_path: Optional path to save the generated plot
        """
        # Load data and extract key components
        df = pd.read_csv(results_csv_path)
        model_names = df["Model_Name"].unique()
        metrics = ["ROUGE1", "ROUGE2", "ROUGEL"]
        score_types = ["P", "R", "F"]

        # Create the plot
        num_models = len(model_names)
        fig, axes = plt.subplots(
            num_models, len(metrics), figsize=(6 * len(metrics), 4 * num_models)
        )

        if num_models == 1:
            axes = np.expand_dims(axes, axis=0)  # ensure 2D indexing

        # Plot the data
        self._plot_clean_unclean_bars(df, model_names, metrics, score_types, axes)

        # Finalize the plot
        handles, labels = axes[-1][-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=6)
        fig.suptitle("Model Performance on Clean vs Unclean Data", fontsize=16)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            self.logger.info("Saved clean vs unclean comparison plot to %s", save_path)

        plt.close()

    def _plot_clean_unclean_bars(self, df, model_names, metrics, score_types, axes):
        """
        Helper method to plot clean vs unclean bars to reduce complexity

        Args:
            df: DataFrame with evaluation results
            model_names: Names of models to display
            metrics: Metrics to show (ROUGE1, ROUGE2, etc.)
            score_types: Score types to display (P, R, F)
            axes: Matplotlib axes to plot on
        """
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


class TokenDataVisualizer:
    """
    Handles all visualizations related to token length analysis
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.TokenDataVisualizer")
        self.logger.setLevel(config.log_level)
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
        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

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
        # Extract stat data once to reduce variable count
        train_d_stats = stats["train"]["dialogues"]
        train_s_stats = stats["train"]["summaries"]
        test_d_stats = stats["test"]["dialogues"]
        test_s_stats = stats["test"]["summaries"]

        # Set up plot
        _fig = plt.figure(figsize=(20, 10))
        metrics = ["mean", "median", "mode", "max", "min", "std"]
        x_positions = np.arange(len(metrics)) * 2
        width = 0.35

        # Create main visualization area
        main_ax = plt.axes([0.1, 0.1, 0.7, 0.8])
        legend_ax = plt.axes([0.82, 0.1, 0.15, 0.8])
        legend_ax.axis("off")

        # Plot the bars
        legend_handles = self._create_stat_bars(
            main_ax,
            metrics,
            x_positions,
            width,
            train_d_stats,
            train_s_stats,
            test_d_stats,
            test_s_stats,
        )

        # Configure axes
        main_ax.set_title(
            "Token Length Statistics Comparison (Values Shown Above Bars)"
        )
        main_ax.set_xticks(x_positions)
        main_ax.set_xticklabels([m.capitalize() for m in metrics])
        main_ax.set_ylabel("Token Count")
        main_ax.grid(axis="y", alpha=0.3)

        # Add legend
        legend_labels = [
            "Train Dialogues",
            "Train Summaries",
            "Test Dialogues",
            "Test Summaries",
        ]
        legend_ax.legend(
            legend_handles, legend_labels, loc="upper left", framealpha=0.7
        )

        # Add stats text box
        stats_text = self._create_stats_text(
            train_d_stats, train_s_stats, test_d_stats, test_s_stats
        )
        legend_ax.text(
            0,
            0.5,
            "\n".join(stats_text),
            ha="left",
            va="center",
            fontfamily="monospace",
            bbox={"facecolor": "white", "alpha": 0.7},
        )

        plt.tight_layout()

        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            self.logger.info("Saved statistics plot to %s", save_path)
        plt.close()

    def _create_stat_bars(
        self,
        ax,
        metrics,
        x_positions,
        width,
        train_d_stats,
        train_s_stats,
        test_d_stats,
        test_s_stats,
    ):
        """
        Helper method to create and label the statistic bars

        Args:
            ax: Matplotlib axis to plot on
            metrics: List of metrics to display
            x_positions: X-axis positions for the bars
            width: Width of each bar
            train_d_stats: Train dialogue statistics
            train_s_stats: Train summary statistics
            test_d_stats: Test dialogue statistics
            test_s_stats: Test summary statistics

        Returns:
            List of legend handles
        """
        legend_handles = []
        y_offset = 0.05

        # Train dialogue bars
        bars = ax.bar(
            x_positions - width / 2,
            [train_d_stats[m] for m in metrics],
            width,
            color=self.color_palette["train"]["dialogues"],
            alpha=0.7,
        )
        legend_handles.append(bars[0])
        self._add_value_labels(ax, bars, [train_d_stats[m] for m in metrics], y_offset)

        # Train summary bars
        bars = ax.bar(
            x_positions + width / 2,
            [train_s_stats[m] for m in metrics],
            width,
            color=self.color_palette["train"]["summaries"],
            alpha=0.7,
        )
        legend_handles.append(bars[0])
        self._add_value_labels(ax, bars, [train_s_stats[m] for m in metrics], y_offset)

        # Test dialogue bars
        bars = ax.bar(
            x_positions - width / 2 + width,
            [test_d_stats[m] for m in metrics],
            width,
            color=self.color_palette["test"]["dialogues"],
            alpha=0.7,
        )
        legend_handles.append(bars[0])
        self._add_value_labels(ax, bars, [test_d_stats[m] for m in metrics], y_offset)

        # Test summary bars
        bars = ax.bar(
            x_positions + width / 2 + width,
            [test_s_stats[m] for m in metrics],
            width,
            color=self.color_palette["test"]["summaries"],
            alpha=0.7,
        )
        legend_handles.append(bars[0])
        self._add_value_labels(ax, bars, [test_s_stats[m] for m in metrics], y_offset)

        return legend_handles

    def _add_value_labels(self, ax, bars, values, y_offset):
        """
        Add value labels above bars

        Args:
            ax: Matplotlib axis to add labels to
            bars: Bar containers from plt.bar
            values: Values to display
            y_offset: Vertical offset for labels
        """
        for j, bar_item in enumerate(bars):
            height = bar_item.get_height()
            ax.text(
                bar_item.get_x() + bar_item.get_width() / 2,
                height + y_offset,
                f"{values[j]:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    def _create_stats_text(
        self, train_d_stats, train_s_stats, test_d_stats, test_s_stats
    ):
        """
        Create text summary of statistics

        Args:
            train_d_stats: Train dialogue statistics
            train_s_stats: Train summary statistics
            test_d_stats: Test dialogue statistics
            test_s_stats: Test summary statistics

        Returns:
            List of strings for text display
        """
        stats_text = ["Statistics Summary:"]

        # Add train set stats
        stats_text.append("\nTrain Set:")
        stats_text.append("Dialogues (Mean / Median / Mode/ Max/ Min/ SD):")
        stats_text.append(
            f"{train_d_stats['mean']:.1f} / "
            f"{train_d_stats['median']:.1f} / "
            f"{train_d_stats['mode']}/ "
            f"{train_d_stats['max']}/ "
            f"{train_d_stats['min']}/ "
            f"{train_d_stats['std']:.1f}"
        )
        stats_text.append("Summaries (Mean / Median / Mode/ Max/ Min/ SD):")
        stats_text.append(
            f"{train_s_stats['mean']:.1f} / "
            f"{train_s_stats['median']:.1f} / "
            f"{train_s_stats['mode']}/ "
            f"{train_s_stats['max']}/ "
            f"{train_s_stats['min']}/ "
            f"{train_s_stats['std']:.1f}"
        )

        # Add test set stats
        stats_text.append("\nTest Set:")
        stats_text.append("Dialogues (Mean / Median / Mode/ Max/ Min/ SD):")
        stats_text.append(
            f"{test_d_stats['mean']:.1f} / "
            f"{test_d_stats['median']:.1f} / "
            f"{test_d_stats['mode']}/ "
            f"{test_d_stats['max']}/ "
            f"{test_d_stats['min']}/ "
            f"{test_d_stats['std']:.1f}"
        )
        stats_text.append("Summaries (Mean / Median / Mode/ Max/ Min/ SD):")
        stats_text.append(
            f"{test_s_stats['mean']:.1f} / "
            f"{test_s_stats['median']:.1f} / "
            f"{test_s_stats['mode']}/ "
            f"{test_s_stats['max']}/ "
            f"{test_s_stats['min']}/ "
            f"{test_s_stats['std']:.1f}"
        )

        return stats_text
