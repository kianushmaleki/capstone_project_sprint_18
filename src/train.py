import mlflow
import mlflow.sklearn
import yaml
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from scipy.sparse import hstack


CLASS_LABELS = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_training_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    tfidf_title = TfidfVectorizer(max_features=3000)
    tfidf_desc = TfidfVectorizer(max_features=5000)

    X_train = hstack([
        tfidf_title.fit_transform(train_df["Title"].fillna("")),
        tfidf_desc.fit_transform(train_df["Description"].fillna("")),
    ])
    X_test = hstack([
        tfidf_title.transform(test_df["Title"].fillna("")),
        tfidf_desc.transform(test_df["Description"].fillna("")),
    ])
    return X_train, X_test, tfidf_title, tfidf_desc


def train(config: dict) -> None:
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    df = load_training_data(f"{config['data']['raw_path']}/train.csv")
    y = df.iloc[:, 0]

    train_df, val_df, y_train, y_val = train_test_split(
        df,
        y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=y,
    )

    X_train, X_val, tfidf_title, tfidf_desc = build_features(train_df, val_df)

    candidates = [
        ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=config["training"]["random_state"])),
        ("RandomForest",       RandomForestClassifier(n_estimators=100, random_state=config["training"]["random_state"])),
        ("NaiveBayes",         MultinomialNB()),
    ]

    best_run_id = None
    best_f1 = -1

    for name, model in candidates:
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            acc = accuracy_score(y_val, preds)
            f1  = f1_score(y_val, preds, average="weighted")

            mlflow.log_param("model", name)
            mlflow.log_params(config["training"])
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_weighted", f1)
            mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"{name:25s}  accuracy={acc:.4f}  f1={f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_run_id = mlflow.active_run().info.run_id

    print(f"\nBest run: {best_run_id}  (f1={best_f1:.4f})")
    return best_run_id


if __name__ == "__main__":
    config = load_config()
    train(config)
