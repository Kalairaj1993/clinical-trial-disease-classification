"""Create exploratory data-analysis tables and plots."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

def run_eda(input_path: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path, low_memory=False)
    required = {"source_condition_query", "brief_summary"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")

    df["brief_summary"] = df["brief_summary"].fillna("").astype(str)
    df["summary_word_count"] = df["brief_summary"].str.split().str.len()
    df["summary_char_count"] = df["brief_summary"].str.len()

    overview = pd.DataFrame({
        "metric": ["rows", "columns", "unique_categories", "missing_summaries",
                   "blank_summaries", "exact_duplicate_rows", "duplicate_nct_ids"],
        "value": [
            len(df), len(df.columns), df["source_condition_query"].nunique(),
            int(df["brief_summary"].isna().sum()),
            int(df["brief_summary"].str.strip().eq("").sum()),
            int(df.duplicated().sum()),
            int(df["nct_id"].duplicated().sum()) if "nct_id" in df else "n/a"
        ]
    })
    overview.to_csv(out / "dataset_overview.csv", index=False)
    counts = df["source_condition_query"].value_counts().rename_axis("category").reset_index(name="count")
    counts.to_csv(out / "category_counts.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    counts.sort_values("count").plot.barh(x="category", y="count", ax=ax, legend=False)
    ax.set_title("Clinical Trial Records by Source Category")
    ax.set_xlabel("Number of records")
    ax.set_ylabel("Source category")
    fig.tight_layout()
    fig.savefig(out / "category_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["summary_word_count"].clip(upper=df["summary_word_count"].quantile(.99)),
            bins=40, edgecolor="black")
    ax.set_title("Brief Summary Length (word count; clipped at 99th percentile)")
    ax.set_xlabel("Words per summary")
    ax.set_ylabel("Number of records")
    fig.tight_layout()
    fig.savefig(out / "summary_length_distribution.png", dpi=160)
    plt.close(fig)

    # Most frequent terms are exploratory only; they are not used to fit the model.
    texts = df["brief_summary"].loc[df["brief_summary"].str.strip().ne("")]
    vectorizer = CountVectorizer(stop_words="english", max_features=30, ngram_range=(1, 1))
    matrix = vectorizer.fit_transform(texts)
    totals = matrix.sum(axis=0).A1
    words = pd.DataFrame({"word": vectorizer.get_feature_names_out(), "count": totals})
    words = words.sort_values("count", ascending=False)
    words.to_csv(out / "top_terms.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    words.sort_values("count").plot.barh(x="word", y="count", ax=ax, legend=False)
    ax.set_title("Most Frequent Terms in Brief Summaries")
    ax.set_xlabel("Term count")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out / "top_terms.png", dpi=160)
    plt.close(fig)

    print(f"EDA outputs written to: {out.resolve()}")
    print(overview.to_string(index=False))
    print("\nCategory counts:\n", counts.to_string(index=False))
    print("\nNote: frequent terms are descriptive, not proof of clinical relevance.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/clinical_trials_cleaned.csv")
    parser.add_argument("--output", default="outputs/eda")
    args = parser.parse_args()
    run_eda(args.input, args.output)
