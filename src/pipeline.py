import logging
from data_loader import DataLoader
from model import SummarizationModel
from evaluator import SummarizationEvaluator
from config import config
import pandas as pd
from visualization import SummaryVisualizer
import os


class SummarizationPipeline:
    """
    Orchestrates the entire summarization process from data loading to evaluation.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(config.LOG_LEVEL)
        self.data_loader = DataLoader()
        self.model = SummarizationModel()
        self.evaluator = SummarizationEvaluator()
        self.visualizer = SummaryVisualizer()
        self.results_filepath = "results/summarization_scores.csv"

        # Initialize results file with header if it doesn't exist
        if not os.path.exists(self.results_filepath):
            os.makedirs(os.path.dirname(self.results_filepath), exist_ok=True)
            pd.DataFrame(
                columns=[
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
            ).to_csv(self.results_filepath, index=False)

    def save_single_result(self, result_row):
        """Save a single result row to the CSV file."""
        flat_row = {
            "Data_Type": result_row["Data_Type"],
            "Model_Name": result_row["Model_Name"],
            "ROUGE1_P": float(result_row.get("rouge1", {}).get("precision", 0)),
            "ROUGE1_R": float(result_row.get("rouge1", {}).get("recall", 0)),
            "ROUGE1_F": float(result_row.get("rouge1", {}).get("fmeasure", 0)),
            "ROUGE2_P": float(result_row.get("rouge2", {}).get("precision", 0)),
            "ROUGE2_R": float(result_row.get("rouge2", {}).get("recall", 0)),
            "ROUGE2_F": float(result_row.get("rouge2", {}).get("fmeasure", 0)),
            "ROUGEL_P": float(result_row.get("rougeL", {}).get("precision", 0)),
            "ROUGEL_R": float(result_row.get("rougeL", {}).get("recall", 0)),
            "ROUGEL_F": float(result_row.get("rougeL", {}).get("fmeasure", 0)),
        }

        df = pd.DataFrame([flat_row])
        df.to_csv(self.results_filepath, mode="a", header=False, index=False)
        self.logger.info(
            f"Appended results for {result_row['Model_Name']} ({result_row['Data_Type']}) to {self.results_filepath}"
        )

    def run(self, sample_size: int = None):
        self.logger.info("Starting summarization pipeline")
        all_results = []

        # Store scores separately for plotting comparisons per data type
        grouped_scores = {"Clean": [], "Unclean": []}

        for clean_flag in [True, False]:
            data_type = "Clean" if clean_flag else "Unclean"
            self.logger.info(f"\n\n--- Running with {data_type} Data ---\n")
            config.CLEAN_DATA = clean_flag
            data = self.data_loader.load_data(sample_size)
            dialogues, summaries = data["test"]

            for model_name in config.MODEL_NAMES:
                self.logger.info("Running for model: %s", model_name)
                config.MODEL_NAME = model_name
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

            self.visualizer.plot_model_comparison(
                grouped_scores[data_type],
                config.MODEL_NAMES,
                save_path=f"plots/{data_type.lower()}/model_comparison.png",
            )

        # Final comparison plot
        self.visualizer.plot_clean_vs_unclean_comparison(
            results_csv_path=self.results_filepath,
            save_path="plots/clean_vs_unclean_model_performance_comparison.png",
        )

        return all_results


if __name__ == "__main__":
    pipeline = SummarizationPipeline()
    calculated_scores = pipeline.run(sample_size=3)
    print("scores:", calculated_scores)
