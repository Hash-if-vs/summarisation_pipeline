import logging
from pipeline import SummarizationPipeline
from config import config


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)


def print_results(results):
    """Print the results in a more readable format."""
    print("\n" + "=" * 80)
    print("SUMMARIZATION MODEL PERFORMANCE RESULTS".center(80))
    print("=" * 80 + "\n")

    for result in results:
        print(f"\nModel: {result['Model_Name']}")
        print(f"Data Type: {result['Data_Type']}")
        print("-" * 60)

        # Print ROUGE-1 scores
        print("ROUGE-1:")
        print(f"  Precision: {result['rouge1']['precision']:.4f}")
        print(f"  Recall:    {result['rouge1']['recall']:.4f}")
        print(f"  F1-Score:  {result['rouge1']['fmeasure']:.4f}")

        # Print ROUGE-2 scores
        print("\nROUGE-2:")
        print(f"  Precision: {result['rouge2']['precision']:.4f}")
        print(f"  Recall:    {result['rouge2']['recall']:.4f}")
        print(f"  F1-Score:  {result['rouge2']['fmeasure']:.4f}")

        # Print ROUGE-L scores
        print("\nROUGE-L:")
        print(f"  Precision: {result['rougeL']['precision']:.4f}")
        print(f"  Recall:    {result['rougeL']['recall']:.4f}")
        print(f"  F1-Score:  {result['rougeL']['fmeasure']:.4f}")
        print("-" * 60)


def main():
    setup_logging()

    try:
        pipeline = SummarizationPipeline()
        results = pipeline.run(sample_size=3)
        print_results(results)
    except Exception as e:
        logging.error("Pipeline failed: %s", str(e))
        raise


if __name__ == "__main__":
    main()
