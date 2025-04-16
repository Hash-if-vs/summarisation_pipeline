import logging
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Model configuration
    MODEL_NAMES: List[str] = field(
        default_factory=lambda: [
            "Hashif/bart-samsum",
            "philschmid/distilbart-cnn-12-6-samsum",
            "philschmid/bart-large-cnn-samsum",
            "sharmax-vikas/flan-t5-base-samsum",
            "google/flan-t5-base",
            "facebook/bart-large",
        ]
    )
    SELECTED_MODEL_INDEX: int = 0
    MAX_INPUT_LENGTH: int = 800
    MAX_OUTPUT_LENGTH: int = 81
    NUM_BEAMS: int = 4

    # Data configuration
    DATASET_NAME: str = "Samsung/samsum"
    TEST_SPLIT_SIZE: float = 0.2
    RANDOM_SEED: int = 42

    # Evaluation configuration
    METRICS: List[str] = field(default_factory=lambda: ["rouge1", "rouge2", "rougeL"])

    # Logging configuration
    LOG_LEVEL: int = logging.INFO
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    CLEAN_DATA: bool = True  # Set to False to disable cleaning

    # This will be set in __post_init__
    MODEL_NAME: str = ""

    def __post_init__(self):
        self.MODEL_NAME = self.MODEL_NAMES[self.SELECTED_MODEL_INDEX]


config = Config()
