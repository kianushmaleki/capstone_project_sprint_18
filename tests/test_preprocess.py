import pytest
import pandas as pd
from unittest.mock import patch
from src.preprocess import load_data, plot_class_distribution


def test_load_data_returns_dataframe(tmp_path):
    csv = tmp_path / "train.csv"
    csv.write_text("class,Title,Description\n1,Hello,World\n2,Foo,Bar\n")
    df = load_data(str(csv))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_load_data_columns(tmp_path):
    csv = tmp_path / "train.csv"
    csv.write_text("class,Title,Description\n1,Hello,World\n")
    df = load_data(str(csv))
    assert list(df.columns) == ["class", "Title", "Description"]


def test_plot_class_distribution_runs_without_error():
    df = pd.DataFrame({
        "class_index": [1, 1, 2, 3, 3, 3, 4],
        "Title":       ["t"] * 7,
        "Description": ["d"] * 7,
    })
    with patch("matplotlib.pyplot.show"):
        plot_class_distribution(df)


def test_plot_class_distribution_all_four_classes():
    df = pd.DataFrame({"class_index": [1, 2, 3, 4]})
    with patch("matplotlib.pyplot.show"):
        # Should not raise even with one sample per class
        plot_class_distribution(df)
