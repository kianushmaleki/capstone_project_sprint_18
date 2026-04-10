import mlflow
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(config: dict) -> None:
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run():
        mlflow.log_params(config["training"])

        # Load data
        df = pd.read_csv(f"{config['data']['processed_path']}/data.csv")

        # Split data
        train_df, test_df = train_test_split(
            df,
            test_size=config["training"]["test_size"],
            random_state=config["training"]["random_state"],
        )

        # Add your model training logic here

        # Log metrics
        # mlflow.log_metric("accuracy", accuracy)

        print("Training complete.")


if __name__ == "__main__":
    config = load_config()
    train(config)
