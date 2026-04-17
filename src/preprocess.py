import pandas as pd
import matplotlib.pyplot as plt


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with nulls in key columns and reset the index."""
    before = len(df)
    df = df.dropna(subset=df.columns.tolist()).reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} rows with missing values ({after} remaining)")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Combine Title and Description into a single text field for richer features."""
    df = df.copy()
    df["text"] = df["Title"].fillna("") + " " + df["Description"].fillna("")
    df["text"] = df["text"].str.lower().str.strip()
    df["title_len"] = df["Title"].fillna("").str.split().str.len()
    df["desc_len"]  = df["Description"].fillna("").str.split().str.len()
    return df


def plot_class_distribution(df: pd.DataFrame) -> None:
    class_labels = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

    counts = df.iloc[:, 0].value_counts().sort_index()
    labels = [class_labels.get(i, str(i)) for i in counts.index]

    _, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts.values, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"], edgecolor="white")

    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{count:,}", ha="center", va="bottom", fontsize=10)

    ax.set_title("Class Index Distribution (AG News)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_ylim(0, counts.max() * 1.12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = load_data("data/raw/train.csv")
    print(data.info())
    print(data.describe())
    print(data.head())
    data = clean_data(data)
    data = feature_engineering(data)
    plot_class_distribution(data)
