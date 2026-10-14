# Clinical Trial Disease Category Classification Using NLP and Machine Learning

An educational end-to-end project that classifies a clinical-trial `brief_summary` into one of the source categories found in the supplied CSV. It uses Pandas, TF-IDF, scikit-learn and Streamlit.

> **Important:** This is not a patient diagnosis system. `source_condition_query` is the dataset's source/search category. A trial may discuss several conditions, and the source category is not guaranteed to be a medically verified single label. Treat outputs as research/educational predictions only.

## 1. Dataset verified

The supplied file `clinical_trials_raw_patient2trial_conditions.csv` was inspected. It has **60,337 rows and 16 columns**:

`source_condition_query`, `nct_id`, `title`, `official_title`, `brief_summary`, `conditions`, `interventions`, `overall_status`, `study_type`, `phase`, `sex`, `minimum_age`, `maximum_age`, `healthy_volunteers`, `eligibility_criteria`, `clinicaltrials_url`.

Initial checks found:
- `brief_summary`: 0 missing values and 0 blank strings.
- Exact duplicate rows: 0.
- Duplicate `nct_id` values: 0.
- `source_condition_query`: 8 categories.
- Missingness in other metadata columns exists (for example, `phase` has many missing values), but those fields are not required for this text-only model.

Category counts observed in the original file:

| Source category | Rows |
|---|---:|
| breast cancer | 16,301 |
| type 2 diabetes | 11,467 |
| covid-19 | 10,153 |
| anxiety | 9,286 |
| chronic obstructive pulmonary disease | 6,181 |
| rheumatoid arthritis | 3,637 |
| glaucoma | 2,173 |
| sickle cell anemia | 1,139 |

The dataset is imbalanced. The training script therefore reports macro and weighted metrics and uses macro F1 to select between the candidate models. Counts can change if the source CSV changes.

## 2. Folder structure

```text
clinical_trial_disease_classification/
├── data/
│   └── clinical_trials_raw_patient2trial_conditions.csv  # add the supplied file here
├── models/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
├── preprocess.py
├── eda.py
├── train_model.py
├── evaluate_model.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

Generated files:
- `data/clinical_trials_cleaned.csv`
- `outputs/eda/` — overview, category counts, frequent terms and charts
- `outputs/model_evaluation/` — comparison table, classification report and confusion matrix
- `models/disease_classifier.joblib` — fitted TF-IDF + classifier pipeline
- `models/model_metadata.json` — model details used by the app

## 3. Setup (Windows)

Open Command Prompt or the VS Code terminal in this project folder.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy the supplied CSV into the `data` folder. Keep the filename exactly:
`data/clinical_trials_raw_patient2trial_conditions.csv`

## 4. Run the pipeline in order

```bash
python preprocess.py
python eda.py
python train_model.py
python evaluate_model.py
streamlit run app.py
```

The first training run can take a few minutes depending on computer RAM and CPU. The app requires the model created by `train_model.py`.

## 5. What each file does

### `preprocess.py`
Loads the CSV, checks required columns, normalizes label case and whitespace, removes records without a usable ID, label or summary, and removes repeated trial IDs/exact duplicates if found. The initial supplied file had no missing/blank brief summaries, no duplicate rows and no duplicate NCT IDs. Text is not aggressively stripped of punctuation or negation because those can matter in medical language.

### `eda.py`
Creates:
- `dataset_overview.csv`
- `category_counts.csv`
- `category_distribution.png`
- `summary_length_distribution.png`
- `top_terms.csv`
- `top_terms.png`

Top terms are exploratory only. They do not establish causation or clinical importance.

### `train_model.py`
- Uses only `brief_summary` as model input and `source_condition_query` as target.
- Splits data into stratified 80% training and 20% test sets with a fixed random seed.
- Fits TF-IDF only on training text through a scikit-learn Pipeline, preventing vocabulary leakage from the test split.
- TF-IDF settings: lowercase, Unicode accent normalization, English stop words, unigrams and bigrams, minimum document frequency 2, maximum 100,000 features, sublinear term frequency.
- Compares LinearSVC (class-balanced) and Multinomial Naive Bayes.
- Selects the model with the highest macro F1 on the held-out test split. Since the test set is also used for model selection, the resulting test score is somewhat optimistic; use a validation split or nested cross-validation for a stronger final estimate.
- Saves the pipeline, comparison metrics, per-class classification report, confusion matrix and metadata.

### `evaluate_model.py`
Recreates the same fixed stratified split and prints/saves classification metrics for the saved model. Use this as a convenient reporting script. It is not an independent external validation set.

### `app.py`
Streamlit interface with:
- Summary text input and predicted source category
- Relative decision scores for LinearSVC, clearly labeled as not probabilities
- Dataset category chart and summary metrics
- Model metadata and limitations

## 6. Understanding the metrics

- **Accuracy:** proportion of all test predictions that are correct.
- **Precision:** among predictions for a category, the fraction that are correct.
- **Recall:** among true examples of a category, the fraction recovered.
- **F1-score:** harmonic mean of precision and recall.
- **Macro F1:** F1 calculated separately per category then averaged equally; useful with imbalanced categories.
- **Weighted F1:** F1 averaged using each category's support.
- **Confusion matrix:** rows are true labels and columns are predicted labels; off-diagonal cells show category confusions.

Do not invent results before running the training script. Copy actual values from `outputs/model_evaluation/model_comparison.csv` and `classification_report.csv` into your final report.

## 7. Important methodology and leakage checks

The labels appear to be the categories used to collect or query trials. This is a usable supervised-learning target for a prototype, but it is not automatically equivalent to a clinically adjudicated disease label. Before claiming deployment readiness:
1. Inspect random examples from every category, including summaries that mention multiple conditions.
2. Check repeated or near-duplicate summaries across categories; the supplied file has no exact duplicate rows, but this does not rule out near duplicates.
3. Consider a group split by trial ID if future versions include repeated versions or multiple rows per trial.
4. Use a validation set or nested cross-validation to select models, and reserve a final untouched test set for final reporting.
5. Test on newer or independently sourced trials, and review errors with a domain expert.
6. Do not use the system for diagnosis, treatment, eligibility decisions or other high-stakes clinical decisions.

## 8. GitHub notes

The `.gitignore` excludes CSV files, saved models and generated outputs because the dataset is about 151 MB and model artifacts may be large. Keep sensitive or licensed data out of public repositories. If you are allowed to share the dataset, upload it separately or provide documented download instructions. Commit the Python source, `requirements.txt`, and README.

## 9. Suggested final report sections

1. Abstract
2. Introduction and problem statement
3. Objectives and scope
4. Dataset description and target-label caveats
5. Data cleaning and exploratory analysis
6. NLP and TF-IDF feature extraction
7. Algorithms and experimental setup
8. Results: accuracy, macro/weighted F1, per-class scores and confusion matrix
9. Streamlit application
10. Limitations, ethical considerations and future work
11. Conclusion and references
