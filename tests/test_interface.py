import pytest
import pandas as pd
from scipy.sparse import issparse
from unittest.mock import MagicMock, patch
from src.app import (
    parse_input,
    predict,
    explain_prediction,
    fit_vectorizers,
    CLASS_LABELS,
    CLASS_EXPLANATIONS,
)
from src.train import load_config


@pytest.fixture
def config():
    return load_config()


# --- parse_input ---

def test_parse_input_splits_on_period():
    result = parse_input("Apple hits record revenue. iPhone sales drive growth.")
    assert result["title"] == "Apple hits record revenue"
    assert result["description"] == "iPhone sales drive growth."


def test_parse_input_single_sentence():
    result = parse_input("Apple hits record revenue")
    assert result["title"] == "Apple hits record revenue"
    assert result["description"] == "Apple hits record revenue"


def test_parse_input_returns_dict_with_keys():
    result = parse_input("Some news article text.")
    assert "title" in result
    assert "description" in result


def test_parse_input_strips_whitespace():
    result = parse_input("  Headline here.  Body text here.  ")
    assert result["title"] == "Headline here"
    assert result["description"] == "Body text here."


# --- explain_prediction ---

def test_explain_prediction_known_labels():
    for label in ["World", "Sports", "Business", "Sci/Tech"]:
        explanation = explain_prediction(label)
        assert isinstance(explanation, str)
        assert len(explanation) > 0


def test_explain_prediction_unknown_label():
    result = explain_prediction("Unknown")
    assert result == "Category could not be explained."


def test_explain_prediction_matches_class_explanations():
    for label, expected in CLASS_EXPLANATIONS.items():
        assert explain_prediction(label) == expected


# --- fit_vectorizers ---

def test_fit_vectorizers_returns_two_vectorizers(config):
    df = pd.DataFrame({
        "Title":       ["world politics", "sports game", "stock market"],
        "Description": ["global news", "team wins", "earnings rise"],
    })
    v1, v2 = fit_vectorizers(df, config)
    assert hasattr(v1, "transform")
    assert hasattr(v2, "transform")


# --- predict ---

def test_predict_returns_valid_class(config):
    df = pd.DataFrame({
        "Title":       ["tech launch", "election", "match result", "gdp growth"],
        "Description": ["new AI model", "voting results", "final score", "economy slows"],
    })
    tfidf_title, tfidf_desc = fit_vectorizers(df, config)

    mock_model = MagicMock()
    mock_model.predict.return_value = [2]

    result = predict("some title", "some description", mock_model, tfidf_title, tfidf_desc)
    assert result == 2
    assert result in CLASS_LABELS


def test_predict_passes_sparse_matrix_to_model(config):
    df = pd.DataFrame({
        "Title":       ["news headline"],
        "Description": ["article body text"],
    })
    tfidf_title, tfidf_desc = fit_vectorizers(df, config)

    mock_model = MagicMock()
    mock_model.predict.return_value = [1]

    predict("news headline", "article body text", mock_model, tfidf_title, tfidf_desc)

    call_args = mock_model.predict.call_args[0][0]
    assert issparse(call_args)
