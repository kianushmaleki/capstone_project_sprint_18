import os
import json
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

load_dotenv()

CLASS_LABELS = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

CLASS_EXPLANATIONS = {
    "World":    "This article discusses international events, geopolitics, or global affairs.",
    "Sports":   "This article covers athletic competitions, teams, players, or sporting events.",
    "Business": "This article relates to markets, companies, economics, or financial activity.",
    "Sci/Tech": "This article involves science, technology, innovation, or research developments.",
}

PARSE_SYSTEM_PROMPT = """You are a feature-extraction assistant for a news classification system.
Given a user's free-text input, extract a news article title and description.
Respond ONLY with a valid JSON object in this exact format:
{
  "title": "<short headline>",
  "description": "<one or two sentence summary>",
  "out_of_scope": false
}
Set "out_of_scope" to true if the input is a personal question, a general chat message,
or anything that is clearly not a news article (e.g. "what is 2+2", "how are you", "tell me a joke").
If the input is too vague to extract a title or description, set both text fields to empty strings."""

EXPLAIN_SYSTEM_PROMPT = """You are a helpful assistant that explains news classification results.
Given an article and its predicted category, write a short 2-3 sentence explanation of why it
likely belongs to that category. Be conversational and concise."""


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


def fit_vectorizers(train_df: pd.DataFrame, config: dict):
    tfidf_title = TfidfVectorizer(max_features=config["tfidf"]["max_features_title"])
    tfidf_desc  = TfidfVectorizer(max_features=config["tfidf"]["max_features_desc"])
    tfidf_title.fit(train_df["Title"].fillna(""))
    tfidf_desc.fit(train_df["Description"].fillna(""))
    return tfidf_title, tfidf_desc


def _llm_parse(user_text: str, client) -> dict:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        system=PARSE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_text}],
    )
    raw = response.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"title": "", "description": "", "out_of_scope": False}


def _rule_parse(user_text: str) -> dict:
    sentences = user_text.strip().split(". ", maxsplit=1)
    title       = sentences[0].strip()
    description = sentences[1].strip() if len(sentences) > 1 else title
    return {"title": title, "description": description}


def parse_input(user_text: str, client=None) -> dict:
    if client is not None:
        return _llm_parse(user_text, client)
    return _rule_parse(user_text)


def _llm_explain(title: str, description: str, label: str, client) -> str:
    prompt = (
        f'Article title: "{title}"\n'
        f'Article description: "{description}"\n'
        f'Predicted category: {label}\n\n'
        "Explain why this article likely belongs to that category."
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        system=EXPLAIN_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def explain_prediction(label: str, title: str = "", description: str = "", client=None) -> str:
    if client is not None:
        return _llm_explain(title, description, label, client)
    return CLASS_EXPLANATIONS.get(label, "Category could not be explained.")


def predict(title: str, description: str, model, tfidf_title, tfidf_desc) -> int:
    X = hstack([
        tfidf_title.transform([title]),
        tfidf_desc.transform([description]),
    ])
    return int(model.predict(X)[0])


def main():
    print("Loading config and model...")
    config = load_config()

    # Use LLM if API key is available, otherwise fall back to rule-based
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client  = None
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            print("LLM mode: active (Anthropic API key detected)")
        except ImportError:
            print("LLM mode: disabled (anthropic package not installed)")
    else:
        print("LLM mode: disabled (no ANTHROPIC_API_KEY found — using rule-based fallback)")

    run_id = find_best_run(config["mlflow"]["experiment_name"])
    model  = load_model(run_id)

    train_df = pd.read_csv(f"{config['data']['raw_path']}/train.csv")
    tfidf_title, tfidf_desc = fit_vectorizers(train_df, config)

    print("\nAG News Classifier — type 'quit' to exit")
    print("Example: \"Apple reports record quarterly earnings driven by iPhone sales\"\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        parsed      = parse_input(user_input, client)
        title       = parsed.get("title", "").strip()
        description = parsed.get("description", "").strip()

        if parsed.get("out_of_scope"):
            print("Assistant: I'm a news classifier — I can only categorise news articles into World, Sports, Business, or Sci/Tech. Try describing a news story.\n")
            continue

        if not title and not description:
            print("Assistant: I couldn't extract enough information. Could you describe the news article in more detail?\n")
            continue

        class_index = predict(title, description, model, tfidf_title, tfidf_desc)
        label       = CLASS_LABELS.get(class_index, "Unknown")
        explanation = explain_prediction(label, title, description, client)

        print(f"\nCategory  : {label}")
        print(f"Assistant : {explanation}\n")


if __name__ == "__main__":
    main()
