"""Streamlit interface for clinical-trial disease-category prediction."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Clinical Trial Disease Classifier", page_icon="🧬", layout="wide")
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "disease_classifier.joblib"
DATA_PATH = ROOT / "data" / "clinical_trials_cleaned.csv"
META_PATH = ROOT / "models" / "model_metadata.json"

st.title("🧬 Clinical Trial Disease Category Classification")
st.caption("NLP + TF-IDF + Machine Learning")
st.warning(
    "Research/education prototype only. This predicts the dataset's source category "
    "from a trial summary; it does not diagnose patients or provide medical advice."
)

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python train_model.py` first.")
    st.stop()

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(str(MODEL_PATH))
metadata = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
tab_predict, tab_dashboard, tab_about = st.tabs(["🔎 Predict category", "📊 Dataset dashboard", "ℹ️ About"])

with tab_predict:
    st.subheader("Enter a clinical trial brief summary")
    sample = st.text_area(
        "Brief Summary",
        height=220,
        placeholder="Paste a clinical trial brief summary here..."
    )
    if st.button("Predict disease category", type="primary"):
        if not sample.strip():
            st.error("Please enter a non-empty clinical trial summary.")
        else:
            prediction = model.predict([sample.strip()])[0]
            st.success(f"Predicted source category: **{prediction.title()}**")
            st.info(
                "This is a model classification, not a calibrated probability. "
                "Review the original trial details before drawing conclusions."
            )
            classifier = model.named_steps.get("classifier")
            if hasattr(classifier, "decision_function"):
                scores = model.decision_function([sample.strip()])
                classes = list(model.classes_)
                if getattr(scores, "ndim", 1) == 1:
                    ranked = sorted(zip(classes, scores), key=lambda x: x[1], reverse=True)[:3]
                else:
                    ranked = sorted(zip(classes, scores[0]), key=lambda x: x[1], reverse=True)[:3]
                st.caption("Top relative decision scores (not probabilities)")
                st.dataframe(
                    pd.DataFrame(ranked, columns=["Category", "Relative decision score"]),
                    use_container_width=True, hide_index=True
                )
            elif hasattr(classifier, "predict_proba"):
                probabilities = model.predict_proba([sample.strip()])[0]
                ranked = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)[:3]
                st.caption("Top model-estimated probabilities (not clinical certainty)")
                st.dataframe(
                    pd.DataFrame(ranked, columns=["Category", "Model probability"]),
                    use_container_width=True, hide_index=True
                )

with tab_dashboard:
    st.subheader("Dataset overview")
    if not DATA_PATH.exists():
        st.info("Place the prepared CSV at `data/clinical_trials_cleaned.csv` to enable dataset charts.")
    else:
        data = pd.read_csv(DATA_PATH, usecols=["source_condition_query", "brief_summary"], low_memory=False)
        counts = data["source_condition_query"].value_counts().rename_axis("Category").reset_index(name="Trials")
        c1, c2, c3 = st.columns(3)
        c1.metric("Clinical trial records", f"{len(data):,}")
        c2.metric("Disease categories", f"{data['source_condition_query'].nunique():,}")
        c3.metric("Blank summaries", f"{data['brief_summary'].fillna('').astype(str).str.strip().eq('').sum():,}")
        fig = px.bar(counts, x="Category", y="Trials", title="Trials by source category", text="Trials")
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(counts, use_container_width=True, hide_index=True)

with tab_about:
    st.subheader("Model information")
    st.write(f"**Selected model:** {metadata.get('selected_model', 'See training report')}")
    st.write(f"**Text field:** `{metadata.get('text_column', 'brief_summary')}`")
    st.write(f"**Target label:** `{metadata.get('target_column', 'source_condition_query')}`")
    st.write(f"**Training rows:** {metadata.get('training_rows', 'Not recorded')}")
    st.write(f"**Held-out test rows:** {metadata.get('test_rows', 'Not recorded')}")
    st.markdown("""
    **Pipeline:** Brief summary → TF-IDF word and two-word features → trained classifier → predicted category.

    The target column is named `source_condition_query`. In this dataset it represents the
    disease-specific source/search category, not necessarily a verified single diagnosis for
    every trial. Performance on this dataset may not generalize to other sources or future trials.
    """)
