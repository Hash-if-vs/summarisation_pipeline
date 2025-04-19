import logging
from dataclasses import dataclass, field
from typing import List, Dict
import os


@dataclass
class Config:
    # Model configuration
    model_names: List[str] = field(
        default_factory=lambda: [
            "Hashif/bart-samsum",
            "philschmid/distilbart-cnn-12-6-samsum",
            "philschmid/bart-large-cnn-samsum",
            "sharmax-vikas/flan-t5-base-samsum",
            "google/flan-t5-base",
            "facebook/bart-large",
        ]
    )
    selected_model_index: int = 0
    max_input_length: int = 800
    max_output_length: int = 81
    num_beams: int = 4

    # Data configuration
    dataset_name: str = "Samsung/samsum"
    test_split_size: float = 0.2
    random_seed: int = 42

    # File paths configuration
    file_save_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "generated_summaries": os.path.join("results", "summaries"),
            "evaluation_scores": os.path.join("results", "metrics", "scores"),
            "text_statistics": os.path.join("results", "metrics", "text_stats"),
            "performance_plots": os.path.join(
                "results", "visualizations", "performance"
            ),
            "text_analysis_plots": os.path.join(
                "results", "visualizations", "text_analysis"
            ),
            "logs": os.path.join("logs"),
            "processed_dataset": os.path.join("data", "processed"),
        }
    )

    # Evaluation configuration
    metrics: List[str] = field(default_factory=lambda: ["rouge1", "rouge2", "rougeL"])

    # Logging configuration
    log_level: int = logging.INFO
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    clean_data: bool = True  # Set to False to disable cleaning

    # This will be set in __post_init__
    model_name: str = ""

    def __post_init__(self):
        self.model_name = self.model_names[self.selected_model_index]
        # Ensure all file save directories exist
        for path in self.file_save_paths.values():
            os.makedirs(path, exist_ok=True)


config = Config()
