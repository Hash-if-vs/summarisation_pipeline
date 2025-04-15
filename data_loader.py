from dotenv import load_dotenv

load_dotenv()
from datasets import load_dataset
from typing import Dict, Tuple, Any
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
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.cleaner = DataCleaner() if config.CLEAN_DATA else None
        if config.CLEAN_DATA:
            self.logger.info("Data cleaning enabled")
        else:
            self.logger.info("Data cleaning disabled")

    def load_data(self) -> Dict[str, Tuple]:
        """
        Load and optionally clean the dataset.
        """
        self.logger.info("Loading dataset: %s", config.DATASET_NAME)

        try:
            dataset = load_dataset(config.DATASET_NAME, trust_remote_code=True)
            raw_data = {
                "train": (dataset["train"]["dialogue"], dataset["train"]["summary"]),
                "test": (dataset["test"]["dialogue"], dataset["test"]["summary"]),
            }

            self.logger.info(
                "Dataset loaded successfully. Train samples: %d, Test samples: %d",
                len(raw_data["train"][0]),
                len(raw_data["test"][0]),
            )

            if config.CLEAN_DATA:
                return {
                    "train": self.cleaner.clean_dataset(raw_data["train"]),
                    "test": self.cleaner.clean_dataset(raw_data["test"]),
                }
            return raw_data

        except Exception as e:
            self.logger.error("Failed to load dataset: %s", str(e))
            raise

    def analyze_dataset(self, data: Dict[str, Tuple]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze token length distribution of the dataset.
        """
        self.logger.info("Analyzing dataset token distribution")

        stats = {}
        for split in ["train", "test"]:
            dialogues, summaries = data[split]

            # Calculate token lengths
            dialogue_lengths = [
                len(self.tokenizer.tokenize(dialogue))
                for dialogue in dialogues[:1000]  # Sample for efficiency
            ]
            summary_lengths = [
                len(self.tokenizer.tokenize(summary)) for summary in summaries[:1000]
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
                f"{split.capitalize()} set analysis:\n"
                f"Dialogues - Mean: {stats[split]['dialogues']['mean']:.1f}, "
                f"Median: {stats[split]['dialogues']['median']}, "
                f"Mode: {stats[split]['dialogues']['mode']}\n"
                f"Summaries - Mean: {stats[split]['summaries']['mean']:.1f}, "
                f"Median: {stats[split]['summaries']['median']}, "
                f"Mode: {stats[split]['summaries']['mode']}"
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
