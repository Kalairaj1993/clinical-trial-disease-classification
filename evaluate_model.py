"""Evaluate the saved model on a reproducible hold-out split."""
from pathlib import Path
import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def main(input_path, model_path, output_path):
    df = pd.read_csv(input_path, low_memory=False)
    df["brief_summary"] = df["brief_summary"].fillna("").astype(str).str.strip()
    df["source_condition_query"] = df["source_condition_query"].astype(str).str.strip().str.lower()
    df = df[(df["brief_summary"] != "") & (df["source_condition_query"] != "")]
    X_train, X_test, y_train, y_test = train_test_split(
        df["brief_summary"], df["source_condition_query"],
        test_size=0.20, random_state=42, stratify=df["source_condition_query"]
    )
    model = joblib.load(model_path)
    predictions = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions, zero_division=0))
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).transpose().to_csv(output_path)
    print(f"Report saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/clinical_trials_cleaned.csv")
    parser.add_argument("--model", default="models/disease_classifier.joblib")
    parser.add_argument("--output", default="outputs/model_evaluation/re_evaluation_report.csv")
    args = parser.parse_args()
    main(args.input, args.model, args.output)
