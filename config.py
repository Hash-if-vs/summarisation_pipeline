import logging
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Model configuration
    MODEL_NAME: str = "philschmid/bart-large-cnn-samsum"
    MAX_INPUT_LENGTH: int = 1024
    MAX_OUTPUT_LENGTH: int = 128
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


config = Config()
