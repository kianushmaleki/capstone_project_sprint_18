import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


CLASS_LABELS = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

CLASS_EXPLANATIONS = {
    "World":    "This article discusses international events, geopolitics, or global affairs.",
    "Sports":   "This article covers athletic competitions, teams, players, or sporting events.",
    "Business": "This article relates to markets, companies, economics, or financial activity.",
    "Sci/Tech": "This article involves science, technology, innovation, or research developments.",
}


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_best_run(experiment_name: str) -> str:
    try:
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            order_by=["metrics.f1_weighted DESC"],
        )
    except Exception:
        runs = pd.DataFrame()

    if runs.empty:
        runs = mlflow.search_runs(
            search_all_experiments=True,
            order_by=["metrics.f1_weighted DESC"],
        )

    if runs.empty:
        raise RuntimeError(
            "\n[ERROR] No trained models found.\n"
            "  Run this first:  python src/train.py\n"
        )
    return runs.iloc[0]["run_id"]


def load_model(run_id: str):
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


def fit_vectorizers(train_df: pd.DataFrame):
    tfidf_title = TfidfVectorizer(max_features=3000)
    tfidf_desc  = TfidfVectorizer(max_features=5000)
    tfidf_title.fit(train_df["Title"].fillna(""))
    tfidf_desc.fit(train_df["Description"].fillna(""))
    return tfidf_title, tfidf_desc


def parse_input(user_text: str) -> dict:
    """Split user text into title (first sentence) and description (remainder)."""
    sentences = user_text.strip().split(". ", maxsplit=1)
    title       = sentences[0].strip()
    description = sentences[1].strip() if len(sentences) > 1 else title
    return {"title": title, "description": description}


def predict(title: str, description: str, model, tfidf_title, tfidf_desc) -> int:
    X = hstack([
        tfidf_title.transform([title]),
        tfidf_desc.transform([description]),
    ])
    return int(model.predict(X)[0])


def explain_prediction(label: str) -> str:
    return CLASS_EXPLANATIONS.get(label, "Category could not be explained.")


def main():
    print("Loading config and model...")
    config = load_config()

    run_id = find_best_run(config["mlflow"]["experiment_name"])
    model  = load_model(run_id)

    train_df = pd.read_csv(f"{config['data']['raw_path']}/train.csv")
    tfidf_title, tfidf_desc = fit_vectorizers(train_df)

    print("AG News Classifier — type 'quit' to exit")
    print("Example: \"Apple reports record quarterly earnings driven by iPhone sales\"\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        parsed      = parse_input(user_input)
        title       = parsed["title"]
        description = parsed["description"]

        class_index = predict(title, description, model, tfidf_title, tfidf_desc)
        label       = CLASS_LABELS.get(class_index, "Unknown")
        explanation = explain_prediction(label)

        print(f"\nCategory  : {label}")
        print(f"Assistant : {explanation}\n")


if __name__ == "__main__":
    main()
