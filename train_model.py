"""Train and compare TF-IDF text classifiers; save the best by macro F1."""
from pathlib import Path
import json
import time
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

TEXT_COL = "brief_summary"
TARGET_COL = "source_condition_query"

def make_pipeline(classifier):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=100_000,
            sublinear_tf=True,
            dtype="float32"
        )),
        ("classifier", classifier)
    ])

def train(input_path="data/clinical_trials_cleaned.csv", output_dir="models",
          reports_dir="outputs/model_evaluation", test_size=0.20, random_state=42):
    out = Path(output_dir)
    reports = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path, low_memory=False)
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str).str.strip()
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().str.lower()
    df = df[(df[TEXT_COL] != "") & (df[TARGET_COL] != "")].copy()

    counts = df[TARGET_COL].value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("Need at least two classes and at least two rows per class.")

    X_train, X_test, y_train, y_test = train_test_split(
        df[TEXT_COL], df[TARGET_COL], test_size=test_size,
        random_state=random_state, stratify=df[TARGET_COL]
    )
    candidates = {
        "LinearSVC": LinearSVC(class_weight="balanced", random_state=random_state),
        "MultinomialNB": MultinomialNB(alpha=0.5)
    }
    results = []
    fitted = {}

    for name, classifier in candidates.items():
        print(f"\nTraining {name}...")
        model = make_pipeline(classifier)
        started = time.time()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        elapsed = time.time() - started
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average="macro", zero_division=0
        )
        weighted_precision, weighted_recall, weighted_f1, _ = (
            precision_recall_fscore_support(
                y_test, predictions, average="weighted", zero_division=0
            )
        )
        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
            "training_seconds": round(elapsed, 2)
        }
        results.append(row)
        fitted[name] = (model, predictions)
        print(f"{name}: accuracy={row['accuracy']:.4f}, macro F1={f1:.4f}")

    results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    results_df.to_csv(reports / "model_comparison.csv", index=False)
    best_name = results_df.iloc[0]["model"]
    best_model, best_predictions = fitted[best_name]
    joblib.dump(best_model, out / "disease_classifier.joblib")

    labels = sorted(df[TARGET_COL].unique())
    report = classification_report(
        y_test, best_predictions, labels=labels, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(reports / "classification_report.csv")
    cm = confusion_matrix(y_test, best_predictions, labels=labels)
    fig, ax = plt.subplots(figsize=(11, 9))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
        ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Confusion Matrix — {best_name}")
    fig.tight_layout()
    fig.savefig(reports / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    metadata = {
        "selected_model": best_name,
        "text_column": TEXT_COL,
        "target_column": TARGET_COL,
        "classes": labels,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": test_size,
        "random_state": random_state,
        "selection_metric": "macro_f1",
        "note": "Scores are dataset-specific; validate on external/newer trials before real-world use."
    }
    with open(out / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSelected model: {best_name} (highest held-out macro F1)")
    print(results_df.to_string(index=False))
    print(f"\nSaved model: {out / 'disease_classifier.joblib'}")
    print(f"Saved reports: {reports.resolve()}")
    return results_df, metadata

if __name__ == "__main__":
    train()
