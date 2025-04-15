# Text Summarization Pipeline

This project provides a modular pipeline for evaluating and comparing different text summarization models using the Hugging Face ecosystem. It allows for data loading, optional cleaning, model inference, evaluation using ROUGE metrics, and visualization of results.

## Features

-   **Multiple Model Comparison**: Evaluate and compare several pre-trained summarization models from Hugging Face Hub (configurable in `config.py`).
-   **Modular Design**: Components for data loading (`data_loader.py`), cleaning (`data_cleaner.py`), model inference (`model.py`), evaluation (`evaluator.py`), and visualization (`visualization.py`) are separated for clarity and extensibility.
-   **Configurable Pipeline**: Centralized configuration (`config.py`) using dataclasses for easy management of model names, generation parameters, dataset details, logging, and processing options.
-   **Data Cleaning Option**: Includes an optional data cleaning step (`data_cleaner.py`) which can be toggled in the configuration.
-   **Comprehensive Evaluation**:
    -   Calculates ROUGE-1, ROUGE-2, and ROUGE-L scores.
    -   Compares model performance on both original ("Unclean") and cleaned ("Clean") data.
    -   Saves detailed scores to a CSV file (`results/summarization_scores.csv`).
-   **Visualization**:
    -   Generates plots comparing model performance for clean and unclean data (`plots/clean/model_comparison.png`, `plots/unclean/model_comparison.png`).
    -   Generates a plot comparing overall performance across models between clean and unclean data (`plots/clean_vs_unclean_model_performance_comparison.png`).
    -   (Optional, if run standalone) Plots token length distributions (`plots/token_distributions.png`).
    -   Displays example dialogue/summary pairs alongside generated summaries.
-   **Production-Ready**: Incorporates logging, error handling, and type hints.

## Architecture

The pipeline follows a clear sequence:

1.  **Configuration (`config.py`)**: Defines all parameters, including models to test, dataset, generation settings, and flags like `CLEAN_DATA`.
2.  **Data Loading (`data_loader.py`)**: Loads the specified dataset (e.g., "Samsung/samsum") using `datasets`. Optionally cleans the data using `DataCleaner`. Can also perform token analysis if run separately.
3.  **Pipeline Orchestration (`pipeline.py`)**:
    -   Iterates through data cleaning options (True/False).
    -   For each option, loads data using `DataLoader`.
    -   Iterates through the list of models defined in `config.MODEL_NAMES`.
    -   Loads the current model using `SummarizationModel`.
    -   Generates summaries for the test set.
    -   Evaluates generated summaries against references using `SummarizationEvaluator` (ROUGE scores).
    -   Uses `SummaryVisualizer` to create comparison plots and display examples.
    -   Collects all results.
4.  **Model Inference (`model.py`)**: Uses `transformers.pipeline` (or `AutoModelForSeq2SeqLM` and `AutoTokenizer`) to load and run the summarization models specified in the config.
5.  **Evaluation (`evaluator.py`)**: Computes ROUGE metrics using the `evaluate` library (likely, based on Hugging Face ecosystem standards).
6.  **Visualization (`visualization.py`)**: Uses libraries like Matplotlib/Seaborn (inferred) to generate plots comparing model performance and display text examples.
7.  **Main Entry Point (`main.py`)**: Sets up logging, instantiates the `SummarizationPipeline`, runs it (optionally with a `sample_size`), and prints the formatted evaluation results to the console.

## Project Structure

```
.
├── .env                  # Environment variables (e.g., HF_TOKEN) - Optional
├── .gitignore
├── config.py             # Central configuration
├── data_cleaner.py       # Text cleaning logic
├── data_loader.py        # Data loading and analysis
├── evaluator.py          # Evaluation metric calculation
├── main.py               # Main script to run the pipeline
├── model.py              # Summarization model loading and inference
├── pipeline.py           # Pipeline orchestration logic
├── requirements.txt      # Project dependencies
├── tuner.py              # (Assumed) Hyperparameter tuning logic
├── visualization.py      # Results visualization and plotting
├── notebooks/            # Directory for experimental notebooks
├── plots/                # Output directory for generated plots
│   ├── clean/
│   │   └── model_comparison.png
│   ├── unclean/
│   │   └── model_comparison.png
│   └── clean_vs_unclean_model_performance_comparison.png
└── results/              # Output directory for evaluation scores
    └── summarization_scores.csv
```

## Setup

### Prerequisites

-   Python 3.8+
-   `pip` package manager
-   Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Hash-if-vs/summarisation_pipeline
    cd summarisation_pipeline
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Optional) Create `.env` file:** If your models require authentication or you use specific environment variables, create a `.env` file in the root directory:
    ```env
    # Example: If using private models
    # HUGGING_FACE_HUB_TOKEN=your_hf_token
    ```
    The `data_loader.py` uses `python-dotenv` to load these variables.

## Configuration

Modify `config.py` to customize the pipeline:

-   `MODEL_NAMES`: List of Hugging Face model identifiers to evaluate.
-   `SELECTED_MODEL_INDEX`: Default model index (primarily used if running `model.py` directly, the pipeline iterates through all).
-   `MAX_INPUT_LENGTH`, `MAX_OUTPUT_LENGTH`, `NUM_BEAMS`: Generation parameters.
-   `DATASET_NAME`: The Hugging Face dataset identifier (e.g., "Samsung/samsum").
-   `CLEAN_DATA`: Set to `True` to enable data cleaning, `False` to disable. The pipeline runs both scenarios by default.
-   `LOG_LEVEL`, `LOG_FORMAT`: Logging configuration.

## Usage

To run the full pipeline evaluating all configured models on both clean and unclean data:

```bash
python main.py
```

The script will:
-   Load the data.
-   Run the pipeline once with `CLEAN_DATA=True`.
-   Run the pipeline again with `CLEAN_DATA=False`.
-   Generate summaries for each model on the test set (or a sample if modified).
-   Calculate and print ROUGE scores for each model and data condition.
-   Save detailed scores to `results/summarization_scores.csv`.
-   Save comparison plots to the `plots/` directory.
-   Print final formatted results to the console.

*Note: The `pipeline.run()` method in `main.py` currently uses a default `sample_size=500`. Modify this in `main.py` if you want to run on the full dataset (set to `None`) or a different sample size.*

## Design Choices

-   **Modularity**: Separating concerns into different files (`data_loader`, `model`, `evaluator`, etc.) makes the codebase easier to understand, maintain, and extend. New models, datasets, or evaluation metrics can be added with minimal changes to the core pipeline.
-   **Configuration-Driven**: Using a central `config.py` with a `dataclass` allows for easy modification of parameters without digging into the code. It also provides type safety.
-   **Hugging Face Ecosystem**: Leveraging `datasets`, `transformers`, and `evaluate` (inferred) simplifies tasks like data loading, model handling, and metric calculation, allowing focus on the pipeline logic.
-   **Comparative Analysis**: The pipeline is explicitly designed to compare multiple models and the effect of data cleaning, providing valuable insights through direct evaluation and visualization.
-   **Reproducibility**: Storing results (CSV) and visualizations (plots) ensures that experiments can be documented and reproduced.

