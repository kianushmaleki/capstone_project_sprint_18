import mlflow
import yaml
import pandas as pd


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate(model, test_df: pd.DataFrame) -> dict:
    # Add your evaluation logic here
    metrics = {}
    return metrics


def load_model(run_id: str):
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    return model


if __name__ == "__main__":
    config = load_config()
    # model = load_model("your_run_id")
    # test_df = pd.read_csv(f"{config['data']['processed_path']}/test.csv")
    # metrics = evaluate(model, test_df)
    # print(metrics)
