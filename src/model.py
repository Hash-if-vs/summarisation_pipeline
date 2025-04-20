"""
Module for text summarization using pretrained transformer models.
This module provides a class for loading and using a summarization model
to generate concise summaries from input texts.
"""

from typing import List
import logging
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from config import config


if torch.cuda.is_available():
    # Set the device to GPU if available
    print("cuda")
else:
    print("cpu")  # Assuming you want to use the first GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class SummarizationModel:
    """
    Handles loading and using the summarization model.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(config.log_level)
        self.model = None
        self.tokenizer = None
        self.summarizer = None

    def load_model(self):
        """
        Load the pre-trained model and tokenizer.
        """
        model_name = config.model_name
        self.logger.info("Loading model: %s", model_name)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            # Create a summarization pipeline
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
            )

            self.logger.info("Model loaded successfully")

        except Exception as e:
            self.logger.error("Failed to load model: %s", str(e))
            raise

    def summarize(self, texts: List[str]) -> List[str]:
        """
        Generate summaries for a list of input texts.

        Args:
            texts: List of input texts to summarize

        Returns:
            List of generated summaries
        """
        if not self.summarizer:
            raise ValueError("Model not loaded. Call load_model() first.")

        self.logger.info("Generating summaries for %d texts", len(texts))

        try:
            summaries = self.summarizer(
                texts,
                max_length=config.max_output_length,
                min_length=30,
                do_sample=False,
                num_beams=config.num_beams,
                truncation=True,
            )

            return [summary["summary_text"] for summary in summaries]

        except Exception as e:
            self.logger.error("Failed to generate summaries: %s", str(e))
            raise


if __name__ == "__main__":
    model = SummarizationModel()
    model.load_model()
    sample_texts = [
        """Amanda: I baked cookies. Do you want some?
Jerry: Sure!
Amanda: I'll bring you tomorrow :-)""",
        """Olivia: Who are you voting for in this election?
Oliver: Liberals as always.
Olivia: Me too!!
Oliver: Great""",
        "Another example of a dialogue that needs summarization.",
    ]
    generated_summaries = model.summarize(sample_texts)
    print("Generated summaries:", generated_summaries)
