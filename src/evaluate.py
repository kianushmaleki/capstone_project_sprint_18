import mlflow
import mlflow.sklearn
import yaml
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from scipy.sparse import hstack
import matplotlib.pyplot as plt


CLASS_LABELS = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(run_id: str):
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    tfidf_title = TfidfVectorizer(max_features=3000)
    tfidf_desc  = TfidfVectorizer(max_features=5000)

    tfidf_title.fit(train_df["Title"].fillna(""))
    tfidf_desc.fit(train_df["Description"].fillna(""))

    X_test = hstack([
        tfidf_title.transform(test_df["Title"].fillna("")),
        tfidf_desc.transform(test_df["Description"].fillna("")),
    ])
    return X_test


def evaluate(model, X_test, y_test: pd.Series) -> dict:
    preds = model.predict(X_test)

    metrics = {
        "accuracy":    accuracy_score(y_test, preds),
        "f1_weighted": f1_score(y_test, preds, average="weighted"),
        "f1_macro":    f1_score(y_test, preds, average="macro"),
    }

    print("\n--- Classification Report ---")
    print(classification_report(y_test, preds, target_names=list(CLASS_LABELS.values())))

    plot_confusion_matrix(y_test, preds)
    return metrics


def plot_confusion_matrix(y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(CLASS_LABELS.values()),
    )
    _, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def find_best_run(experiment_name: str) -> str:
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["metrics.f1_weighted DESC"],
    )
    if runs.empty:
        raise ValueError(f"No runs found for experiment '{experiment_name}'")
    best = runs.iloc[0]
    print(f"Best run: {best['run_id']}  (f1_weighted={best['metrics.f1_weighted']:.4f})")
    return best["run_id"]


if __name__ == "__main__":
    config = load_config()

    run_id = find_best_run(config["mlflow"]["experiment_name"])
    model  = load_model(run_id)

    train_df = pd.read_csv(f"{config['data']['raw_path']}/train.csv")
    test_df  = pd.read_csv(f"{config['data']['raw_path']}/test.csv")
    y_test   = test_df.iloc[:, 0]

    X_test  = build_features(train_df, test_df)
    metrics = evaluate(model, X_test, y_test)

    print("\n--- Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
