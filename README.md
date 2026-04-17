# Capstone Project: End-to-End Intelligent ML Application

## Overview

This project is an end-to-end intelligent application that bridges traditional machine learning with a natural language interface. The system consists of a trained predictive model (Classification/Regression) and an LLM-powered interface that allows users to interact with the model using plain English.

The project emphasizes production-level thinking, including experiment tracking with MLflow, automated testing, and modular software architecture.

---

## Project Requirements

### 1. Data Preprocessing and Model Training
- **Cleaning:** Handled missing values, categorical encoding, and feature scaling.
- **Training:** Compared at least three configurations (e.g., Random Forest, XGBoost, or Neural Networks).
- **Evaluation:** Models evaluated on a held-out test set using task-appropriate metrics (Accuracy, F1, RMSE, etc.).

### 2. Experiment Tracking (MLflow)
The training workflow is fully integrated with **MLflow** to ensure reproducibility:
- All hyperparameters, data versions, and evaluation metrics are logged per run.
- The best-performing model is saved as a logged artifact.
- `mlflow.search_runs()` is used to programmatically identify the best run.

### 3. LLM-Powered Interface
The application features a natural language layer built using the **Anthropic API**:
- **Input Parsing:** The LLM extracts structured features from raw user text.
- **Model Invocation:** The system loads the best-trained model to run inference on parsed data.
- **Response Generation:** The LLM explains the prediction in a conversational format.
- **Edge Case Handling:** Identifies incomplete queries and asks clarifying questions.

### 4. Testing Suite
Robustness is ensured via `pytest`:
- **Preprocessing Tests:** Validate missing value handling and feature scaling.
- **Model Tests:** Verify output shape, type, and performance thresholds.
- **Interface Tests:** Ensure the LLM correctly parses natural language into model-ready features.

---

## Repository Structure

```
capstone_project_sprint_18/
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Containerization setup
├── .env.example                # Template for API keys (never commit .env)
├── configs/
│   └── config.yaml             # Training hyperparameters and settings
├── src/
│   ├── preprocess.py           # Data cleaning and feature engineering
│   ├── train.py                # Model training with MLflow integration
│   ├── evaluate.py             # Model evaluation utilities
│   └── app.py                  # LLM interface application
├── tests/
│   ├── test_preprocess.py      # Unit tests for data pipeline
│   ├── test_model.py           # Validation tests for ML model
│   └── test_interface.py       # Tests for LLM parsing and logic
├── notebooks/
│   └── exploration.ipynb       # EDA and experimentation
└── data/
    └── .gitkeep                # Data directory (not committed to Git)
```

---

## Getting Started

### Prerequisites
- Python 3.9+
- API key for your LLM provider (stored in `.env`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-project.git
   cd your-project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env and add your ANTHROPIC_API_KEY
   ```

---

## Usage

**Preprocess data:**
```bash
python src/preprocess.py
```

**Train models and log to MLflow:**
```bash
python src/train.py
```

**View experiment results:**
```bash
mlflow ui
```

**Run the natural language interface:**
```bash
python src/app.py
```

**Run the test suite:**
```bash
pytest tests/ -v
```

---

## Results Summary

The best-performing model achieved an Accuracy/R² of `0.XX`. Detailed logs and experiment comparisons can be viewed via the MLflow UI after running `mlflow ui`.

---

## Reflection

**Challenges:** *(Briefly describe a technical hurdle, e.g., prompt engineering for reliable feature extraction from ambiguous user input.)*

**Learnings:** *(Highlight a key takeaway from the MLOps or LLM integration process.)*

**Future Work:** *(Mention potential improvements, e.g., a web-based UI, streaming responses, or model retraining on new data.)*


## Dataset

The dataset used in this project is the **AG News Topic Classification Dataset**, sourced from [Kaggle](https://www.kaggle.com/).
Link to data : https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset


### Origin

AG News is a collection of more than 1 million news articles gathered from over 2,000 news sources by [ComeToMyHead](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html), an academic news search engine active since July 2004. The dataset is provided by the academic community for research purposes in data mining, information retrieval, and related non-commercial activities.

The topic classification dataset was constructed by Xiang Zhang (xiang.zhang@nyu.edu) and used as a text classification benchmark in:

> Xiang Zhang, Junbo Zhao, Yann LeCun. *Character-level Convolutional Networks for Text Classification.* Advances in Neural Information Processing Systems 28 (NIPS 2015).

### Description

The dataset is built from the 4 largest topic classes in the original corpus:

| Split | Samples per class | Total samples |
|-------|------------------|---------------|
| Train | 30,000           | 120,000       |
| Test  | 1,900            | 7,600         |

### Classes

| Index | Label |
|-------|-------|
| 1     | World |
| 2     | Sports |
| 3     | Business |
| 4     | Sci/Tech |

### File Format

| File | Description |
|------|-------------|
| `train.csv` | Training samples (120,000 rows) |
| `test.csv` | Test samples (7,600 rows) |
| `classes.txt` | Label index to class name mapping |

Each CSV contains 3 columns: **class index** (1–4), **title**, and **description**. Text fields are double-quoted; internal double quotes are escaped as `""`. Newlines within fields are escaped as `\n`.

> **Note:** Data files are not committed to this repository. Download the dataset from Kaggle and place the files under `data/raw/`.