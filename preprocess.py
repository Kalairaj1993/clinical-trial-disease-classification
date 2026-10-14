"""Validate and prepare the raw clinical-trial CSV."""
from pathlib import Path
import argparse
import pandas as pd

REQUIRED_COLUMNS = ["source_condition_query", "nct_id", "brief_summary"]

def prepare_data(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Columns found: {df.columns.tolist()}"
        )

    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("\nMissing values in required columns:")
    print(df[REQUIRED_COLUMNS].isna().sum())

    # Keep the original label text, but normalize case and extra whitespace.
    df["source_condition_query"] = (
        df["source_condition_query"].astype("string").str.strip().str.lower()
    )
    # Text normalization is deliberately conservative: preserve medical words,
    # punctuation, negation and numbers. TF-IDF performs tokenization later.
    df["brief_summary"] = (
        df["brief_summary"].fillna("").astype(str)
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df["nct_id"] = df["nct_id"].astype("string").str.strip()

    before = len(df)
    df = df.dropna(subset=["source_condition_query", "nct_id"])
    df = df[
        df["source_condition_query"].ne("")
        & df["nct_id"].ne("")
        & df["brief_summary"].ne("")
    ].copy()
    removed_empty = before - len(df)

    # IDs should identify trials. Remove repeated IDs after reporting them.
    repeated_ids = int(df["nct_id"].duplicated().sum())
    if repeated_ids:
        df = df.drop_duplicates(subset=["nct_id"], keep="first").copy()

    # Also guard against exact duplicate rows.
    exact_duplicates = int(df.duplicated().sum())
    if exact_duplicates:
        df = df.drop_duplicates().copy()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nRemoved rows with missing/blank required fields: {removed_empty:,}")
    print(f"Repeated NCT IDs found and removed: {repeated_ids:,}")
    print(f"Exact duplicate rows found and removed: {exact_duplicates:,}")
    print(f"Prepared data saved to: {output_path}")
    print(f"Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("\nTarget category counts:")
    print(df["source_condition_query"].value_counts())
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/clinical_trials_raw_patient2trial_conditions.csv")
    parser.add_argument("--output", default="data/clinical_trials_cleaned.csv")
    args = parser.parse_args()
    prepare_data(args.input, args.output)
