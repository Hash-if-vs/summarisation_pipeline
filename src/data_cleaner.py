"""Module for cleaning text data using regex, emoji, contractions, and spaCy."""

import logging
import re
from typing import List, Tuple

import emoji
import spacy
from contractions import fix
from config import config


class DataCleaner:
    """
    Handles advanced cleaning operations for text data using spaCy.
    """

    def __init__(self):
        """Initialize the DataCleaner, loading spaCy model."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(config.log_level)
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            self.logger.info("Initialized DataCleaner with spaCy processor")
        except OSError:
            self.logger.error(
                "Spacy model 'en_core_web_sm' not found. "
                "Download it by running 'python -m spacy download en_core_web_sm'"
            )

    def clean_dialogue(self, dialogue: str) -> str:
        """
        Perform comprehensive cleaning on dialogue text.
        """
        # Fast regex operations
        dialogue = re.sub(r"<.*?>", "", dialogue)  # Remove HTML tags
        dialogue = re.sub(r"\\r\\n", " ", dialogue)
        dialogue = re.sub(r"\s+", " ", dialogue).strip()

        # Text normalization
        dialogue = dialogue.lower()
        dialogue = fix(dialogue)  # Expand contractions
        dialogue = emoji.demojize(dialogue)  # Convert emojis to text

        # Linguistic processing (skip if text is too long or spacy failed)
        # Limit length check to avoid excessive processing time/memory
        if self.nlp and len(dialogue) < 1000:
            doc = self.nlp(dialogue)
            # Lemmatize and remove punctuation
            dialogue = " ".join([token.lemma_ for token in doc if not token.is_punct])

        return dialogue

    def clean_summary(self, summary: str) -> str:
        """
        Perform lighter cleaning on summary text to preserve quality.
        """
        summary = re.sub(r"\s+", " ", summary).strip()
        summary = re.sub(r"\\r\\n", " ", summary)
        summary = fix(summary)
        return summary

    def clean_dataset(
        self, data: Tuple[List[str], List[str]]
    ) -> Tuple[List[str], List[str]]:
        """
        Clean an entire dataset split (dialogues and summaries).
        """
        dialogues, summaries = data

        self.logger.info("Cleaning dataset (size: %d)", len(dialogues))
        cleaned_dialogues = [self.clean_dialogue(d) for d in dialogues]
        cleaned_summaries = [self.clean_summary(s) for s in summaries]

        # Log sample cleaning results if logger level allows
        if self.logger.isEnabledFor(logging.DEBUG) and dialogues:
            self.logger.debug("Original dialogue sample: %s...", dialogues[0][:100])
            self.logger.debug(
                "Cleaned dialogue sample: %s...", cleaned_dialogues[0][:100]
            )

        return cleaned_dialogues, cleaned_summaries
