from datasets import load_dataset
from typing import Dict, Tuple, Any, List
import logging
import numpy as np
from collections import Counter
from transformers import AutoTokenizer
from config import config
from data_cleaner import DataCleaner
from visualization import TokenDataVisualizer
import pandas as pd
import os


class DataLoader:
    """
    Handles loading, preparation, and analysis of the summarization dataset.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(config.LOG_LEVEL)
        self.cleaner = DataCleaner() if config.CLEAN_DATA else None
        if config.CLEAN_DATA:
            self.logger.info("Data cleaning enabled")
        else:
            self.logger.info("Data cleaning disabled")

    def load_data(self, sample_size: int = None) -> Dict[str, Tuple]:
        """
        Load and optionally clean a subset of the dataset based on the provided sample size.
        """
        self.logger.info("Loading dataset: %s", config.DATASET_NAME)

        try:
            dataset = load_dataset(config.DATASET_NAME, trust_remote_code=True)

            # If sample_size is provided, load only the subset of the data
            if sample_size:
                data = {
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
                data = {
                    "train": (
                        dataset["train"]["dialogue"],
                        dataset["train"]["summary"],
                    ),
                    "test": (dataset["test"]["dialogue"], dataset["test"]["summary"]),
                }

            self.logger.info(
                "Dataset loaded successfully. Train samples: %d, Test samples: %d",
                len(data["train"][0]),
                len(data["test"][0]),
            )

            # Optionally clean the data
            if config.CLEAN_DATA:
                data["train"] = self.cleaner.clean_dataset(data["train"])
                data["test"] = self.cleaner.clean_dataset(data["test"])

            return data

        except Exception as e:
            self.logger.error("Failed to load dataset: %s", str(e))
            raise

    def clean_subset(
        self, dialogues: List[str], summaries: List[str]
    ) -> Tuple[List[str], List[str]]:
        if not self.cleaner:
            return dialogues, summaries
        self.logger.info("Cleaning sample of size %d", len(dialogues))
        return self.cleaner.clean_dataset((dialogues, summaries))

    def analyze_dataset(
        self, data: Dict[str, Tuple], label: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze token length distribution of the dataset and save statistics to a shared CSV file.
        """
        self.logger.info("Analyzing dataset token distribution (%s)", label)
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

        stats = {}
        rows = []

        for split in ["train", "test"]:
            dialogues, summaries = data[split]

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

            stats[split] = {
                "dialogues": dialogue_stats,
                "summaries": summary_stats,
            }

            # Add to CSV row format with label
            rows.append(
                [
                    label,
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
                    label,
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
                label,
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
        csv_path = "results/dataset_stats.csv"
        if os.path.exists(csv_path):
            df.to_csv(
                csv_path, mode="a", index=False, header=False
            )  # append without header
        else:
            df.to_csv(csv_path, index=False)  # create new with header

        self.logger.info("Saved token stats to dataset_stats.csv (%s)", label)

        return stats


if __name__ == "__main__":
    for clean_flag in [False, True]:
        config.CLEAN_DATA = clean_flag
        data_type = "clean" if clean_flag else "unclean"
        print(f"\nProcessing {data_type} data...")

        # Load and analyze data
        data_loader = DataLoader()
        loaded_data = data_loader.load_data(sample_size=1000)
        computed_stats = data_loader.analyze_dataset(loaded_data, data_type)

        # Save visualizations
        visualizer = TokenDataVisualizer()
        visualizer.plot_token_distributions(
            computed_stats, save_path=f"plots/{type}/{type}_token_distributions.png"
        )
        visualizer.plot_statistics(
            computed_stats,
            save_path=f"plots/{data_type}/{data_type}_statistical_comparison.png",
        )

        # Sample preview and stats
        print(f"\nSample dialogue ({data_type}):", loaded_data["train"][0][0])
        print(f"Sample summary ({data_type}):", loaded_data["train"][1][0])
        print(f"\nTraining set statistics ({data_type}):", computed_stats["train"])
