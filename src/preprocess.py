import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Add your cleaning logic here
    df = df.dropna()
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Add your feature engineering logic here
    return df


def preprocess(input_path: str, output_path: str) -> None:
    df = load_data(input_path)
    df = clean_data(df)
    df = feature_engineering(df)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    config = load_config()
    preprocess(
        input_path=f"{config['data']['raw_path']}/data.csv",
        output_path=f"{config['data']['processed_path']}/data.csv",
    )
