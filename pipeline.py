import logging
from data_loader import DataLoader
from model import SummarizationModel
from evaluator import SummarizationEvaluator
from config import config
import pandas as pd
from visualization import SummaryVisualizer


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

    def save_scores_to_csv(self, results, filepath):
        flattened_results = []

        for row in results:
            flat_row = {"Data_Type": row["Data_Type"], "Model_Name": row["Model_Name"]}
            for metric in ["rouge1", "rouge2", "rougeL"]:
                score_dict = row.get(metric, {})
                flat_row[f"{metric.upper()}_P"] = float(score_dict.get("precision", 0))
                flat_row[f"{metric.upper()}_R"] = float(score_dict.get("recall", 0))
                flat_row[f"{metric.upper()}_F"] = float(score_dict.get("fmeasure", 0))
            flattened_results.append(flat_row)

        df = pd.DataFrame(flattened_results)
        df.to_csv(filepath, index=False)
        self.logger.info(f"Saved flattened results to {filepath}")

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

            self.visualizer.plot_model_comparison(
                grouped_scores[data_type],
                config.MODEL_NAMES,
                save_path=f"plots/{data_type.lower()}/model_comparison.png",
            )

        # Save final combined results
        self.save_scores_to_csv(all_results, "results/summarization_scores.csv")
        self.visualizer.plot_clean_vs_unclean_comparison(
            results_csv_path="results/summarization_scores.csv",
            save_path="plots/clean_vs_unclean_model_performance_comparison.png",
        )

        return all_results


if __name__ == "__main__":
    pipeline = SummarizationPipeline()
    calculated_scores = pipeline.run(sample_size=3)
    print("scores:", calculated_scores)
