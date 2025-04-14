# Modular Text Summarization Pipeline

A modular, class-based implementation of a text summarization pipeline using self-hosted LLMs.

## Features

- Modular architecture with separate components for data loading, modeling, and evaluation
- Uses DistilBART-CNN (a smaller version of BART) for summarization
- Evaluates summaries using ROUGE metrics
- Handles dialogue-style summarization with the SAMSum dataset

## Installation

1. Clone this repository
2. Create and activate a virtual environment
3. Install dependencies:

```bash
pip install -r requirements.txt

