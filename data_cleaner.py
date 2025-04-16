import re
import emoji
import spacy
from typing import List, Tuple
from contractions import fix
import logging


class DataCleaner:
    """
    Handles advanced cleaning operations for text data using spaCy.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.logger.info("Initialized DataCleaner with spaCy processor")

    def clean_dialogue(self, dialogue: str) -> str:
        """
        Perform comprehensive cleaning on dialogue text.
        """
        # Fast regex operations
        dialogue = re.sub(r"<.*?>", "", dialogue)
        dialogue = re.sub(r"\\r\\n", " ", dialogue)
        dialogue = re.sub(r"\s+", " ", dialogue).strip()

        # Text normalization
        dialogue = dialogue.lower()
        dialogue = fix(dialogue)
        dialogue = emoji.demojize(dialogue)

        # Linguistic processing (skip if text is too long)
        if len(dialogue) < 1000:  # Prevent memory issues
            doc = self.nlp(dialogue)
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

        # Log sample cleaning results
        self.logger.debug("Original dialogue sample: %s...", dialogues[0][:100])
        self.logger.debug("Cleaned dialogue sample: %s...", cleaned_dialogues[0][:100])

        return cleaned_dialogues, cleaned_summaries
