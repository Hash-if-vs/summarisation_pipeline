from dotenv import load_dotenv

load_dotenv()
from datasets import load_dataset
from typing import Dict, Tuple, Any, List
import logging
import numpy as np
from collections import Counter
from transformers import AutoTokenizer
from config import config
from data_cleaner import DataCleaner
from visualization import TokenDataVisualizer


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

    def analyze_dataset(self, data: Dict[str, Tuple]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze token length distribution of the dataset.
        """
        self.logger.info("Analyzing dataset token distribution")
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

        stats = {}
        for split in ["train", "test"]:
            dialogues, summaries = data[split]

            # Calculate token lengths
            dialogue_lengths = [
                len(tokenizer.tokenize(dialogue))
                for dialogue in dialogues[:1000]  # Sample for efficiency
            ]
            summary_lengths = [
                len(tokenizer.tokenize(summary)) for summary in summaries[:1000]
            ]

            stats[split] = {
                "dialogues": {
                    "mean": np.mean(dialogue_lengths),
                    "median": np.median(dialogue_lengths),
                    "mode": Counter(dialogue_lengths).most_common(1)[0][0],
                    "max": max(dialogue_lengths),
                    "min": min(dialogue_lengths),
                    "std": np.std(dialogue_lengths),
                    "values": dialogue_lengths,
                },
                "summaries": {
                    "mean": np.mean(summary_lengths),
                    "median": np.median(summary_lengths),
                    "mode": Counter(summary_lengths).most_common(1)[0][0],
                    "max": max(summary_lengths),
                    "min": min(summary_lengths),
                    "std": np.std(summary_lengths),
                    "values": summary_lengths,
                },
            }

            self.logger.info(
                "%s set analysis:\n"
                "Dialogues - Mean: %.1f, Median: %d, Mode: %d\n"
                "Summaries - Mean: %.1f, Median: %d, Mode: %d",
                split.capitalize(),
                stats[split]["dialogues"]["mean"],
                stats[split]["dialogues"]["median"],
                stats[split]["dialogues"]["mode"],
                stats[split]["summaries"]["mean"],
                stats[split]["summaries"]["median"],
                stats[split]["summaries"]["mode"],
            )

        return stats


if __name__ == "__main__":
    data_loader = DataLoader()
    data = data_loader.load_data()
    stats = data_loader.analyze_dataset(data)

    visualizer = TokenDataVisualizer()
    visualizer.plot_token_distributions(
        stats, save_path="plots/token_distributions.png"
    )
    visualizer.plot_statistics(stats, save_path="plots/statistical_comparison.png")

    print("\nSample dialogue:", data["train"][0][0])
    print("Sample summary:", data["train"][1][0])
    print("\nTraining set statistics:", stats["train"])
