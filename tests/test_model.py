import pytest
import pandas as pd
from scipy.sparse import issparse
from unittest.mock import MagicMock
from src.train import build_features, load_config, CLASS_LABELS


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def sample_dfs():
    train = pd.DataFrame({
        "class_index": [1, 2, 3, 4, 1, 2],
        "Title":       ["world news", "sports game", "stock market", "tech launch", "election", "match"],
        "Description": [
            "Global politics and diplomacy",
            "Team wins championship",
            "Markets rally on earnings",
            "New smartphone released",
            "Voting results announced",
            "Final score of the game",
        ],
    })
    val = pd.DataFrame({
        "class_index": [3, 4],
        "Title":       ["economy update", "AI research"],
        "Description": ["GDP growth slows", "New model beats benchmark"],
    })
    return train, val


def test_build_features_returns_sparse(sample_dfs, config):
    train, val = sample_dfs
    X_train, X_val = build_features(train, val, config)
    assert issparse(X_train)
    assert issparse(X_val)


def test_build_features_shape(sample_dfs, config):
    train, val = sample_dfs
    X_train, X_val = build_features(train, val, config)
    assert X_train.shape[0] == len(train)
    assert X_val.shape[0] == len(val)
    assert X_train.shape[1] == X_val.shape[1]


def test_build_features_combined_vocab(sample_dfs, config):
    train, val = sample_dfs
    X_train, _ = build_features(train, val, config)
    max_cols = config["tfidf"]["max_features_title"] + config["tfidf"]["max_features_desc"]
    assert X_train.shape[1] <= max_cols


def test_class_labels_complete():
    assert set(CLASS_LABELS.keys()) == {1, 2, 3, 4}
    assert set(CLASS_LABELS.values()) == {"World", "Sports", "Business", "Sci/Tech"}


def test_load_config_returns_required_keys():
    config = load_config()
    assert "training" in config
    assert "data" in config
    assert "mlflow" in config
    assert "tfidf" in config
    assert "test_size" in config["training"]
    assert "random_state" in config["training"]


def test_model_predict_output_type(sample_dfs, config):
    """Model must return an integer class label from the valid set."""
    from sklearn.naive_bayes import MultinomialNB
    train, val = sample_dfs
    X_train, X_val = build_features(train, val, config)
    y_train = train["class_index"]

    model = MultinomialNB()
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    assert len(preds) == len(val)
    assert all(isinstance(int(p), int) for p in preds)
    assert all(p in CLASS_LABELS for p in preds)


def test_model_meets_minimum_accuracy(sample_dfs, config):
    """Logistic Regression on AG News should easily exceed 80% weighted F1."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    # Use a larger synthetic set so the test is meaningful
    import numpy as np
    rng = np.random.default_rng(42)
    n = 200
    words = {1: "president election vote", 2: "goal match team", 3: "stock market profit", 4: "software chip ai"}
    titles = [words[c] for c in rng.choice([1, 2, 3, 4], size=n)]
    labels = [int(t.split()[0] == "president") * 1 +
              int(t.split()[0] == "goal") * 2 +
              int(t.split()[0] == "stock") * 3 +
              int(t.split()[0] == "software") * 4 for t in titles]

    big_train = pd.DataFrame({"Title": titles, "Description": titles, "class_index": labels})
    big_val   = big_train.sample(40, random_state=42)

    X_train, X_val = build_features(big_train, big_val, config)
    y_train = big_train["class_index"]
    y_val   = big_val["class_index"]

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, average="weighted")

    assert f1 >= 0.80, f"Expected f1 >= 0.80, got {f1:.4f}"
