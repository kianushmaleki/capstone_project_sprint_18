import pytest
import pandas as pd
from src.preprocess import clean_data, feature_engineering


def test_clean_data_drops_nulls():
    df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
    result = clean_data(df)
    assert result.isnull().sum().sum() == 0
    assert len(result) == 2


def test_feature_engineering_returns_dataframe():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = feature_engineering(df)
    assert isinstance(result, pd.DataFrame)
