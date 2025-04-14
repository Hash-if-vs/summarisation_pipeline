from datasets import load_dataset
from typing import Dict, Tuple
import logging
from config import config


class DataLoader:
    """
    Handles loading and preparation of the summarization dataset.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(config.LOG_LEVEL)

    def load_data(self) -> Dict[str, Tuple]:
        """
        Load and split the dataset into train and test sets.

        Returns:
            Dict containing 'train' and 'test' splits with dialogues and summaries
        """
        self.logger.info("Loading dataset: %s", config.DATASET_NAME)

        try:
            # Load the dataset from HuggingFace
            dataset = load_dataset(config.DATASET_NAME, trust_remote_code=True)

            self.logger.info(
                "Dataset loaded successfully.Test samples: %d",
                len(dataset),
            )

            return {
                "train": (
                    dataset["train"]["dialogue"],
                    dataset["train"]["summary"],
                ),
                "test": (
                    dataset["test"]["dialogue"],
                    dataset["test"]["summary"],
                ),
            }

        except Exception as e:
            self.logger.error("Failed to load dataset: %s", str(e))
            raise


if __name__ == "__main__":
    data_loader = DataLoader()
    data = data_loader.load_data()
    print("Data loaded successfully:", data["train"][0][:2])  # Print first two samples
