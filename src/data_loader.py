"""
Module for loading, preparing and analyzing summarization datasets.
This module provides functionality to load datasets from various sources,
clean the data if needed, and perform statistical analysis.
"""

import logging
import os
from collections import Counter
from typing import Dict, Tuple, Any, List

# Third-party imports
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer


# Local application imports
from config import config
from .data_cleaner import DataCleaner
from .visualization import TokenDataVisualizer


class DataLoader:
    """
    Handles loading, preparation, and analysis of the summarization dataset.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(config.log_level)
        self.cleaner = DataCleaner() if config.clean_data else None

        # File paths and names
        self.dataset_stats_path = os.path.join(
            config.file_save_paths["text_statistics"], "dataset_stats.csv"
        )
        self.token_dist_filename = "token_distributions.png"
        self.stats_comp_filename = "statistical_comparison.png"

        if config.clean_data:
            self.logger.info("Data cleaning enabled")
        else:
            self.logger.info("Data cleaning disabled")

    def load_data(self, sample_size: int = None) -> Dict[str, Tuple]:
        """
        Load and optionally clean a subset of the dataset based on the provided sample size.
        """
        self.logger.info("Loading dataset: %s", config.dataset_name)

        try:
            dataset = load_dataset(config.dataset_name, trust_remote_code=True)

            # If sample_size is provided, load only the subset of the data
            if sample_size:
                dataset_data = {
                    "train": (
                        dataset["train"]["dialogue"][:sample_size],
                        dataset["train"]["summary"][:sample_size],
                    ),
                    "test": (
                        dataset["test"]["dialogue"][:sample_size],
                        dataset["test"]["summary"][:sample_size],
                    ),
                }
            else:
                # Load full dataset if no sample size is provided
                dataset_data = {
                    "train": (
                        dataset["train"]["dialogue"],
                        dataset["train"]["summary"],
                    ),
                    "test": (dataset["test"]["dialogue"], dataset["test"]["summary"]),
                }

            self.logger.info(
                "Dataset loaded successfully. Train samples: %d, Test samples: %d",
                len(dataset_data["train"][0]),
                len(dataset_data["test"][0]),
            )

            # Optionally clean the data
            if config.clean_data:
                dataset_data["train"] = self.cleaner.clean_dataset(
                    dataset_data["train"]
                )
                dataset_data["test"] = self.cleaner.clean_dataset(dataset_data["test"])

            return dataset_data

        except Exception as e:
            self.logger.error("Failed to load dataset: %s", str(e))
            raise

    def clean_subset(
        self, dialogues: List[str], summaries: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Clean a subset of dialogues and summaries using the DataCleaner.

        Args:
            dialogues: List of dialogue texts to clean
            summaries: List of summary texts to clean

        Returns:
            Tuple of cleaned dialogues and summaries
        """
        if not self.cleaner:
            return dialogues, summaries
        self.logger.info("Cleaning sample of size %d", len(dialogues))
        return self.cleaner.clean_dataset((dialogues, summaries))

    def analyze_dataset(
        self, dataset_data: Dict[str, Tuple], data_label: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze token length distribution of the dataset and save statistics to a shared CSV file.
        """
        self.logger.info("Analyzing dataset token distribution (%s)", data_label)
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        dataset_stats = {}
        rows = []

        for split in ["train", "test"]:
            dialogues, summaries = dataset_data[split]

            # Tokenize and compute lengths
            dialogue_lengths = [len(tokenizer.tokenize(d)) for d in dialogues]
            summary_lengths = [len(tokenizer.tokenize(s)) for s in summaries]

            # Compute stats
            dialogue_stats = {
                "mean": np.mean(dialogue_lengths),
                "median": np.median(dialogue_lengths),
                "mode": Counter(dialogue_lengths).most_common(1)[0][0],
                "max": max(dialogue_lengths),
                "min": min(dialogue_lengths),
                "std": np.std(dialogue_lengths),
                "values": dialogue_lengths,
            }
            summary_stats = {
                "mean": np.mean(summary_lengths),
                "median": np.median(summary_lengths),
                "mode": Counter(summary_lengths).most_common(1)[0][0],
                "max": max(summary_lengths),
                "min": min(summary_lengths),
                "std": np.std(summary_lengths),
                "values": summary_lengths,
            }

            dataset_stats[split] = {
                "dialogues": dialogue_stats,
                "summaries": summary_stats,
            }

            # Add to CSV row format with label
            rows.append(
                [
                    data_label,
                    split,
                    "dialogues",
                    dialogue_stats["mean"],
                    dialogue_stats["median"],
                    dialogue_stats["mode"],
                    dialogue_stats["max"],
                    dialogue_stats["min"],
                    dialogue_stats["std"],
                ]
            )
            rows.append(
                [
                    data_label,
                    split,
                    "summaries",
                    summary_stats["mean"],
                    summary_stats["median"],
                    summary_stats["mode"],
                    summary_stats["max"],
                    summary_stats["min"],
                    summary_stats["std"],
                ]
            )

            self.logger.info(
                "%s set analysis (%s):\n"
                "Dialogues - Mean: %.1f, Median: %d, Mode: %d\n"
                "Summaries - Mean: %.1f, Median: %d, Mode: %d",
                split.capitalize(),
                data_label,
                dialogue_stats["mean"],
                dialogue_stats["median"],
                dialogue_stats["mode"],
                summary_stats["mean"],
                summary_stats["median"],
                summary_stats["mode"],
            )

        # Save or append to CSV
        df = pd.DataFrame(
            rows,
            columns=[
                "Label",
                "Split",
                "Section",
                "Mean",
                "Median",
                "Mode",
                "Max",
                "Min",
                "Std Dev",
            ],
        )

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.dataset_stats_path), exist_ok=True)

        if os.path.exists(self.dataset_stats_path):
            df.to_csv(
                self.dataset_stats_path, mode="a", index=False, header=False
            )  # append without header
        else:
            df.to_csv(self.dataset_stats_path, index=False)  # create new with header

        self.logger.info(
            "Saved token stats to %s (%s)", self.dataset_stats_path, data_label
        )

        return dataset_stats

    def get_visualization_path(self, data_label: str, filename: str) -> str:
        """
        Generate a proper path for visualization files based on configuration.

        Args:
            data_label: The data type label (e.g., 'clean', 'unclean')
            filename: The name of the file to save

        Returns:
            Complete file path for saving visualizations
        """
        return os.path.join(
            config.file_save_paths["text_analysis_plots"], data_label, filename
        )


if __name__ == "__main__":
    for clean_flag in [False, True]:
        config.clean_data = clean_flag
        DATA_LABEL = "clean" if clean_flag else "unclean"
        print(f"\nProcessing {DATA_LABEL} data...")

        # Load and analyze data
        data_loader = DataLoader()
        loaded_data = data_loader.load_data(sample_size=1000)
        analysis_stats = data_loader.analyze_dataset(loaded_data, DATA_LABEL)

        # Save visualizations
        visualizer = TokenDataVisualizer()
        visualizer.plot_token_distributions(
            analysis_stats,
            save_path=data_loader.get_visualization_path(
                DATA_LABEL, data_loader.token_dist_filename
            ),
        )
        visualizer.plot_statistics(
            analysis_stats,
            save_path=data_loader.get_visualization_path(
                DATA_LABEL, data_loader.stats_comp_filename
            ),
        )

        # Sample preview and stats
        print(f"\nSample dialogue ({DATA_LABEL}):", loaded_data["train"][0][0])
        print(f"Sample summary ({DATA_LABEL}):", loaded_data["train"][1][0])
        print(f"\nTraining set statistics ({DATA_LABEL}):", analysis_stats["train"])
