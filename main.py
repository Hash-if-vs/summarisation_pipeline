import logging
from pipeline import SummarizationPipeline
from config import config


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)


def main():
    """Main entry point for the summarization pipeline."""
    setup_logging()

    try:
        pipeline = SummarizationPipeline()

        # Run with a small sample size for demonstration
        # Remove sample_size parameter to process all test data
        pipeline.run(sample_size=10)

    except Exception as e:
        logging.error("Pipeline failed: %s", str(e))
        raise


if __name__ == "__main__":
    main()
