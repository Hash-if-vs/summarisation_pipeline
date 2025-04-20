"""
Orchestrates the summarization pipeline from data loading to evaluation
and visualization.
"""

import os
import logging
from typing import Dict, List, Optional, Any

import pandas as pd
from config import config

from .data_loader import DataLoader
from .model import SummarizationModel
from .evaluator import SummarizationEvaluator
from .visualization import SummaryVisualizer


class SummarizationPipeline:
    """
    Orchestrates the entire summarization process from data loading to evaluation.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(config.log_level)
        self.data_loader = DataLoader()
        self.model = SummarizationModel()
        self.evaluator = SummarizationEvaluator()
        self.visualizer = SummaryVisualizer()
        self.results_filepath = os.path.join(
            config.file_save_paths["evaluation_scores"], "summarization_scores.csv"
        )

        # Initialize results file with header if it doesn't exist
        if not os.path.exists(self.results_filepath):
            os.makedirs(os.path.dirname(self.results_filepath), exist_ok=True)
            cols = [
                "Data_Type",
                "Model_Name",
                "ROUGE1_P",
                "ROUGE1_R",
                "ROUGE1_F",
                "ROUGE2_P",
                "ROUGE2_R",
                "ROUGE2_F",
                "ROUGEL_P",
                "ROUGEL_R",
                "ROUGEL_F",
            ]
            pd.DataFrame(columns=cols).to_csv(self.results_filepath, index=False)

    def save_single_result(self, result_row: Dict[str, Any]) -> None:
        """Save a single result row to the CSV file."""
        r1 = result_row.get("rouge1", {})
        r2 = result_row.get("rouge2", {})
        rl = result_row.get("rougeL", {})

        flat_row = {
            "Data_Type": result_row["Data_Type"],
            "Model_Name": result_row["Model_Name"],
            "ROUGE1_P": float(r1.get("precision", 0)),
            "ROUGE1_R": float(r1.get("recall", 0)),
            "ROUGE1_F": float(r1.get("fmeasure", 0)),
            "ROUGE2_P": float(r2.get("precision", 0)),
            "ROUGE2_R": float(r2.get("recall", 0)),
            "ROUGE2_F": float(r2.get("fmeasure", 0)),
            "ROUGEL_P": float(rl.get("precision", 0)),
            "ROUGEL_R": float(rl.get("recall", 0)),
            "ROUGEL_F": float(rl.get("fmeasure", 0)),
        }

        df = pd.DataFrame([flat_row])
        df.to_csv(self.results_filepath, mode="a", header=False, index=False)
        self.logger.info(
            "Appended results for %s (%s) to %s",
            result_row["Model_Name"],
            result_row["Data_Type"],
            self.results_filepath,
        )

    def run(self, sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Run the complete summarization pipeline.

        Args:
            sample_size: Optional size of data sample to process

        Returns:
            List of result dictionaries containing evaluation metrics
        """
        self.logger.info("Starting summarization pipeline")
        all_results = []

        # Store scores separately for plotting comparisons per data type
        grouped_scores: Dict[str, List[Dict[str, Any]]] = {"Clean": [], "Unclean": []}

        for clean_flag in [True, False]:
            data_type = "Clean" if clean_flag else "Unclean"
            self.logger.info("\n\n--- Running with %s Data ---\n", data_type)
            config.clean_data = clean_flag
            data = self.data_loader.load_data(sample_size)
            dialogues, summaries = data["test"]

            for model_name in config.model_names:
                self.logger.info("Running for model: %s", model_name)
                config.model_name = model_name
                self.model.load_model()

                # Generate summaries
                generated = self.model.summarize(dialogues)

                # Evaluate
                scores = self.evaluator.evaluate_batch(summaries, generated)
                self.evaluator.print_evaluation(scores)

                # Visualizations
                self.visualizer.display_examples(
                    dialogues, summaries, generated, data_type, sample_size
                )
                result_row = {
                    "Data_Type": data_type,
                    "Model_Name": model_name,
                    **scores,
                }
                all_results.append(result_row)
                grouped_scores[data_type].append(scores)

                # Save results immediately after each model iteration
                self.save_single_result(result_row)

            # Create path for model comparison visualization
            model_comparison_path = os.path.join(
                config.file_save_paths["performance_plots"],
                data_type.lower(),
                "model_comparison.png",
            )
            # Ensure directory exists
            os.makedirs(os.path.dirname(model_comparison_path), exist_ok=True)

            self.visualizer.plot_model_comparison(
                grouped_scores[data_type],
                config.model_names,
                save_path=model_comparison_path,
            )

        # Final comparison plot
        base_path = config.file_save_paths["performance_plots"]
        comparison_filename = "clean_vs_unclean_model_performance_comparison.png"
        comparison_plot_path = os.path.join(base_path, comparison_filename)
        os.makedirs(os.path.dirname(comparison_plot_path), exist_ok=True)

        self.visualizer.plot_clean_vs_unclean_comparison(
            results_csv_path=self.results_filepath,
            save_path=comparison_plot_path,
        )

        return all_results


if __name__ == "__main__":
    pipeline = SummarizationPipeline()
    calculated_scores = pipeline.run(sample_size=3)
    print("scores:", calculated_scores)
