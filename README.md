# AG News Classifier: End-to-End Intelligent ML Application

## Overview

This project is an end-to-end intelligent application that classifies news articles into four categories — **World**, **Sports**, **Business**, and **Sci/Tech** — using the [AG News dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset).

Users interact with the system in plain English. They describe a news story and the application parses their input, runs it through a trained ML classifier, and returns a category with a natural language explanation. For example:

> **You:** "Apple just announced a new chip that outperforms all competitors in benchmark tests."
> **Assistant:** Category: Sci/Tech — This article involves science, technology, innovation, or research developments.

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

## Architecture Overview

```
User (plain English)
        │
        ▼
  parse_input()          ← LLM (Anthropic API) if key present,
        │                   rule-based split otherwise
        ▼
 TF-IDF Vectorizer       ← fit on training set (Title + Description)
        │                   combined via scipy sparse hstack
        ▼
  Best MLflow Model      ← loaded via mlflow.search_runs() → best f1_weighted
        │
        ▼
  Class Index (1–4)
        │
        ▼
 explain_prediction()    ← LLM generates contextual explanation,
        │                   or static template if no API key
        ▼
  Category + Explanation
```

The ML model and LLM are deliberately decoupled: the model handles the classification, and the LLM handles language understanding and explanation. This means the classifier can be swapped out without changing the interface layer.

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

Five model configurations were trained and tracked with MLflow on the AG News dataset (120,000 training samples, 4 classes):

| Model | Accuracy | F1 Weighted | F1 Macro | Precision |
|---|---|---|---|---|
| **LogisticRegression (C=1.0)** ✓ | **0.9077** | **0.9075** | **0.9075** | **0.9074** |
| LogisticRegression (C=0.1) | 0.9003 | 0.9000 | 0.9000 | 0.9000 |
| LinearSVC (C=1.0) | 0.9048 | 0.9046 | 0.9046 | 0.9045 |
| RandomForest (100 trees) | 0.8656 | 0.8650 | 0.8650 | 0.8653 |
| NaiveBayes | 0.8929 | 0.8925 | 0.8925 | 0.8924 |

**Best model: LogisticRegression (C=1.0)** — selected by highest weighted F1 score (0.9075) via `mlflow.search_runs()`. Logistic Regression outperforms the other configurations on this task because TF-IDF produces high-dimensional sparse linear features that are well-suited to linear classifiers. LinearSVC is a close second (F1 0.9046), while RandomForest underperforms as expected on sparse text data due to its sensitivity to feature redundancy.

Detailed logs and experiment comparisons can be viewed via the MLflow UI:
```bash
mlflow ui
```

---

## Reflection

**Challenges:** The main technical challenge was ensuring the TF-IDF vectorizers fitted on training data were reused consistently across training, evaluation, and the app — without data leakage from the test set. Structuring the pipeline so the same vocabulary is applied at inference time required careful separation of `fit_transform` (training only) from `transform` (validation/test/app).

**Learnings:** MLflow experiment tracking is invaluable for comparing models objectively. Using `mlflow.search_runs()` to programmatically rank runs by a target metric eliminates manual comparison and makes the best-model selection fully reproducible. Making the LLM interface optional (falling back to rule-based logic when no API key is present) also improved the robustness of the application.

**Future Work:** Potential improvements include: (1) a lightweight web UI using Streamlit or Gradio, (2) fine-tuning a small transformer (e.g. DistilBERT) on AG News for higher accuracy, (3) adding hyperparameter search with MLflow's `mlflow.sklearn.autolog()`, and (4) streaming LLM responses for a more interactive experience.


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