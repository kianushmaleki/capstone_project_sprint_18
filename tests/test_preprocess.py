import pytest
import pandas as pd
from unittest.mock import patch
from src.preprocess import load_data, clean_data, feature_engineering, plot_class_distribution


# --- load_data ---

def test_load_data_returns_dataframe(tmp_path):
    csv = tmp_path / "train.csv"
    csv.write_text("class,Title,Description\n1,Hello,World\n2,Foo,Bar\n")
    df = load_data(str(csv))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_load_data_preserves_columns(tmp_path):
    csv = tmp_path / "train.csv"
    csv.write_text("class,Title,Description\n1,Hello,World\n")
    df = load_data(str(csv))
    assert list(df.columns) == ["class", "Title", "Description"]


# --- clean_data ---

def test_clean_data_drops_null_rows():
    df = pd.DataFrame({
        "class":       [1, None, 3],
        "Title":       ["t1", "t2", "t3"],
        "Description": ["d1", "d2", "d3"],
    })
    result = clean_data(df)
    assert len(result) == 2
    assert result.isnull().sum().sum() == 0


def test_clean_data_resets_index():
    df = pd.DataFrame({
        "class":       [1, None, 3],
        "Title":       ["t1", "t2", "t3"],
        "Description": ["d1", "d2", "d3"],
    })
    result = clean_data(df)
    assert list(result.index) == list(range(len(result)))


def test_clean_data_no_nulls_returns_same_length():
    df = pd.DataFrame({
        "class":       [1, 2, 3],
        "Title":       ["t1", "t2", "t3"],
        "Description": ["d1", "d2", "d3"],
    })
    result = clean_data(df)
    assert len(result) == 3


def test_clean_data_does_not_mutate_original():
    df = pd.DataFrame({
        "class":       [1, None],
        "Title":       ["t1", "t2"],
        "Description": ["d1", "d2"],
    })
    original_len = len(df)
    clean_data(df)
    assert len(df) == original_len


# --- feature_engineering ---

def test_feature_engineering_returns_dataframe():
    df = pd.DataFrame({
        "class":       [1, 2],
        "Title":       ["Breaking news", "Sports update"],
        "Description": ["World events", "Game results"],
    })
    result = feature_engineering(df)
    assert isinstance(result, pd.DataFrame)


def test_feature_engineering_adds_text_column():
    df = pd.DataFrame({
        "Title":       ["Hello"],
        "Description": ["World"],
    })
    result = feature_engineering(df)
    assert "text" in result.columns
    assert "hello world" in result["text"].values


def test_feature_engineering_adds_length_columns():
    df = pd.DataFrame({
        "Title":       ["one two three"],
        "Description": ["a b"],
    })
    result = feature_engineering(df)
    assert "title_len" in result.columns
    assert "desc_len" in result.columns
    assert result["title_len"].iloc[0] == 3
    assert result["desc_len"].iloc[0] == 2


def test_feature_engineering_does_not_mutate_original():
    df = pd.DataFrame({
        "Title":       ["Hello"],
        "Description": ["World"],
    })
    original_cols = list(df.columns)
    feature_engineering(df)
    assert list(df.columns) == original_cols


# --- plot_class_distribution ---

def test_plot_class_distribution_runs_without_error():
    df = pd.DataFrame({"class_index": [1, 1, 2, 3, 3, 3, 4]})
    with patch("matplotlib.pyplot.show"):
        plot_class_distribution(df)


def test_plot_class_distribution_all_four_classes():
    df = pd.DataFrame({"class_index": [1, 2, 3, 4]})
    with patch("matplotlib.pyplot.show"):
        plot_class_distribution(df)
