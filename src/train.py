import mlflow
import mlflow.sklearn
import yaml
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score
from scipy.sparse import hstack


CLASS_LABELS = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_training_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_features(train_df: pd.DataFrame, val_df: pd.DataFrame, config: dict):
    tfidf_title = TfidfVectorizer(max_features=config["tfidf"]["max_features_title"])
    tfidf_desc  = TfidfVectorizer(max_features=config["tfidf"]["max_features_desc"])

    X_train = hstack([
        tfidf_title.fit_transform(train_df["Title"].fillna("")),
        tfidf_desc.fit_transform(train_df["Description"].fillna("")),
    ])
    X_val = hstack([
        tfidf_title.transform(val_df["Title"].fillna("")),
        tfidf_desc.transform(val_df["Description"].fillna("")),
    ])

    print(f"Feature vector length: {X_train.shape[1]}")
    return X_train, X_val


def train(config: dict) -> str:
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    data = load_training_data(f"{config['data']['raw_path']}/train.csv")
    y    = data.iloc[:, 0]

    train_df, val_df, y_train, y_val = train_test_split(
        data, y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=y,
    )

    X_train, X_val = build_features(train_df, val_df, config)

    # 5 distinct configurations
    candidates = [
        ("LogisticRegression_C1",  LogisticRegression(C=config["training"]["lr_C"],   max_iter=1000, random_state=config["training"]["random_state"])),
        ("LogisticRegression_C01", LogisticRegression(C=0.1,                          max_iter=1000, random_state=config["training"]["random_state"])),
        ("RandomForest",           RandomForestClassifier(n_estimators=config["training"]["rf_n_estimators"], random_state=config["training"]["random_state"])),
        ("LinearSVC",              LinearSVC(C=config["training"]["svc_C"],            max_iter=2000, random_state=config["training"]["random_state"])),
        ("NaiveBayes",             MultinomialNB()),
    ]

    best_run_id = None
    best_f1     = -1

    for name, model in candidates:
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            acc       = accuracy_score(y_val, preds)
            f1_w      = f1_score(y_val, preds, average="weighted")
            f1_macro  = f1_score(y_val, preds, average="macro")
            precision = precision_score(y_val, preds, average="weighted")

            mlflow.log_param("model", name)
            mlflow.log_param("tfidf_max_features_title", config["tfidf"]["max_features_title"])
            mlflow.log_param("tfidf_max_features_desc",  config["tfidf"]["max_features_desc"])
            mlflow.log_params(config["training"])
            mlflow.log_metric("accuracy",          acc)
            mlflow.log_metric("f1_weighted",       f1_w)
            mlflow.log_metric("f1_macro",          f1_macro)
            mlflow.log_metric("precision_weighted", precision)
            mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"{name:25s}  accuracy={acc:.4f}  f1_weighted={f1_w:.4f}  f1_macro={f1_macro:.4f}  precision={precision:.4f}")

            if f1_w > best_f1:
                best_f1     = f1_w
                best_run_id = mlflow.active_run().info.run_id

    print(f"\nBest run : {best_run_id}")
    print(f"Best f1  : {best_f1:.4f}")
    return best_run_id


if __name__ == "__main__":
    config = load_config()
    train(config)
