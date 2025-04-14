import logging
from data_loader import DataLoader
from model import SummarizationModel
from evaluator import SummarizationEvaluator
from config import config

# from tuner import HyperparameterTuner
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
        self.visaulizer = SummaryVisualizer()

    def run(self, sample_size: int = None):
        """
        Run the complete summarization pipeline with optional tuning

        Args:
            sample_size: Number of samples to process
            tune_params: Whether to perform hyperparameter tuning
        """
        self.logger.info("Starting summarization pipeline")

        # Load data
        data = self.data_loader.load_data()
        test_dialogues, test_summaries = data["test"]

        if sample_size and sample_size < len(test_dialogues):
            test_dialogues = test_dialogues[:sample_size]
            test_summaries = test_summaries[:sample_size]

        # Load model
        self.model.load_model()

        # Generate summaries
        generated_summaries = self.model.summarize(test_dialogues)

        # Evaluate results
        scores = self.evaluator.evaluate_batch(test_summaries, generated_summaries)
        self.evaluator.print_evaluation(scores)
        self.visaulizer.display_examples(
            test_dialogues, test_summaries, generated_summaries
        )
        self.visaulizer.plot_rouge_scores(scores, "plot/rouge_scores.png")

        return scores

    # ... (previous code remains the same)


if __name__ == "__main__":
    pipeline = SummarizationPipeline()
    calculated_scores = pipeline.run(sample_size=5)
    print("scores:", calculated_scores)
