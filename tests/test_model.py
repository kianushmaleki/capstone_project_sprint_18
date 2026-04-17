import pytest
import pandas as pd
from scipy.sparse import issparse
from src.train import build_features, load_config, CLASS_LABELS


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


def test_build_features_returns_sparse(sample_dfs):
    train, val = sample_dfs
    X_train, X_val = build_features(train, val)
    assert issparse(X_train)
    assert issparse(X_val)


def test_build_features_shape(sample_dfs):
    train, val = sample_dfs
    X_train, X_val = build_features(train, val)
    assert X_train.shape[0] == len(train)
    assert X_val.shape[0] == len(val)
    assert X_train.shape[1] == X_val.shape[1]


def test_build_features_combined_vocab(sample_dfs):
    train, val = sample_dfs
    X_train, _ = build_features(train, val)
    # max_features=3000 (title) + max_features=5000 (desc) = 8000 max columns
    assert X_train.shape[1] <= 8000


def test_class_labels_complete():
    assert set(CLASS_LABELS.keys()) == {1, 2, 3, 4}
    assert set(CLASS_LABELS.values()) == {"World", "Sports", "Business", "Sci/Tech"}


def test_load_config_returns_required_keys():
    config = load_config()
    assert "training" in config
    assert "data" in config
    assert "mlflow" in config
    assert "test_size" in config["training"]
    assert "random_state" in config["training"]
