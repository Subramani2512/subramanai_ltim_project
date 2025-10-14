# app/app.py
import json
import os
from io import StringIO
import re

import joblib
import matplotlib.pyplot as plt
import numpy as pd
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv

from llm_suggestions import generate as generate_suggestions

load_dotenv()

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

MODEL_PATH = os.path.join("models", "model.joblib")
META_PATH = os.path.join("models", "metadata.json")

# Fixed human-friendly choices as requested
SEX_OPTIONS = ["Male", "Female"]
CP_OPTIONS = ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"]
ECG_OPTIONS = ["Normal", "ST-T abnormality", "Left ventricular hypertrophy"]
SLOPE_OPTIONS = ["Upsloping", "Flat", "Downsloping"]
THAL_OPTIONS = ["Normal", "Fixed defect", "Reversible defect"]
SMOKING_OPTIONS = ["Never", "Former smoker", "Current smoker"]
YES_NO = ["Yes", "No"]

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta

def _is_numeric_like(values):
    if not values:
        return False
    pat = re.compile(r"^\s*-?\d+(\.\d+)?\s*$")
    hits = [bool(pat.match(str(v))) for v in values]
    return all(hits)

def _map_value(col, human_value, meta_categories):
    """
    Map human-friendly UI values to training codes if metadata suggests numeric coding,
    otherwise return the human value directly.
    """
    hv = str(human_value)

    # If training categories already look textual (not numeric-like), pass-through
    if meta_categories and not _is_numeric_like(meta_categories):
        return hv

    # Numeric-like categories → map to classic UCI encodings
    if col == "Sex":
        return "1" if hv.lower().startswith("m") else "0"

    if col == "Exercise Induced Angina":
        return "1" if hv.lower().startswith("y") else "0"

    if col == "Angeo person or not":
        return "1" if hv.lower().startswith("y") else "0"

    if col == "Chest Pain Type":
        enc = {
            "typical angina": "1",
            "atypical angina": "2",
            "non-anginal pain": "3",
            "asymptomatic": "4",
        }
        return enc.get(hv.lower(), hv)

    if col == "Resting ECG Results":
        enc = {
            "normal": "0",
            "st-t abnormality": "1",
            "left ventricular hypertrophy": "2",
        }
        return enc.get(hv.lower(), hv)

    if col == "Slope of ST Segment":
        enc = {
            "upsloping": "1",
            "flat": "2",
            "downsloping": "3",
        }
        return enc.get(hv.lower(), hv)

    if col == "Thallium Stress Test Result (thal)":
        # classic UCI enc: 3=normal, 6=fixed, 7=reversible
        enc = {
            "normal": "3",
            "fixed defect": "6",
            "reversible defect": "7",
        }
        return enc.get(hv.lower(), hv)

    if col == "Smoking Status":
        # try 0/1 or 0/1/2 depending on training categories length
        if meta_categories and len(meta_categories) == 2:
            # binary (e.g., 0 Non-smoker, 1 Smoker)
            return "1" if hv.lower().startswith("current") else "0"
        # ternary: 0 never, 1 former, 2 current
        enc = {
            "never": "0",
            "former smoker": "1",
            "current smoker": "2",
        }
        return enc.get(hv.lower(), hv)

    # Default: return as-is
    return hv

def prepare_for_model(human_inputs: dict, meta: dict) -> dict:
    """
    Convert the human-friendly UI dict to the exact categories/codes expected by the trained model.
    We use metadata["categories"] to decide whether to map to numeric codes.
    """
    categories = meta.get("categories", {})
    out = {}

    # Numeric fields: pass-through
    for k in [
        "Age", "Resting Blood Pressure", "Total Cholesterol", "Fasting Blood Sugar",
        "Maximum Heart Rate Achieved", "ST Depression (oldpeak)",
        "Number of Major Vessels", "Body Mass Index (BMI)"
    ]:
        out[k] = human_inputs.get(k)

    # Categorical fields: map if needed
    for k in [
        "Sex", "Chest Pain Type", "Resting ECG Results", "Exercise Induced Angina",
        "Slope of ST Segment", "Thallium Stress Test Result (thal)",
        "Smoking Status", "Angeo person or not"
    ]:
        out[k] = _map_value(k, human_inputs.get(k), categories.get(k, []))

    return out

def feature_importance_plot(meta):
    fi = meta.get("feature_importances", [])[:20]
    if not fi:
        st.info("Feature importances not available.")
        return
    names = [d["feature"] for d in fi][::-1]
    vals = [d["importance"] for d in fi][::-1]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(names, vals)
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances")
    st.pyplot(fig)

def main():
    st.title("❤️ Heart Disease Prediction (Random Forest + Groq Suggestions)")
    st.caption("Numeric age; controlled clinical categories; Groq-powered summary, immediate actions, lifestyle tips, and more.")

    if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
        st.error("Model or metadata missing. Please run `python src/train.py` first.")
        st.stop()

    model, meta = load_artifacts()
    num_cols = meta["numeric_columns"]
    cat_cols = meta["categorical_columns"]
    categories = meta.get("categories", {})

    st.sidebar.subheader("Decision Threshold")
    threshold = st.sidebar.slider("Classify as 'risk' at probability ≥", 0.10, 0.90, 0.50, 0.01)
    st.sidebar.caption("Move right for higher precision; left for higher recall.")

    st.subheader("Enter Patient Data")
    tab1, tab2 = st.tabs(["🧍 Manual Form", "📄 CSV Upload"])

    # Manual form
    with tab1:
        c1, c2 = st.columns(2)
        inputs = {}

        with c1:
            inputs["Age"] = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
            inputs["Resting Blood Pressure"] = st.number_input("Resting Blood Pressure (mmHg)", 60, 240, 130)
            inputs["Total Cholesterol"] = st.number_input("Total Cholesterol (mg/dL)", 100, 600, 200)
            inputs["Fasting Blood Sugar"] = st.number_input("Fasting Blood Sugar (mg/dL)", 50, 400, 95)
            inputs["Maximum Heart Rate Achieved"] = st.number_input("Maximum Heart Rate Achieved", 60, 230, 150)
            inputs["ST Depression (oldpeak)"] = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
            inputs["Number of Major Vessels"] = st.number_input("Number of Major Vessels (0–4)", 0, 4, 0)
            inputs["Body Mass Index (BMI)"] = st.number_input("Body Mass Index (BMI)", 10.0, 60.0, 25.0, step=0.1)

        with c2:
            inputs["Sex"] = st.selectbox("Sex", SEX_OPTIONS)
            inputs["Chest Pain Type"] = st.selectbox("Chest Pain Type", CP_OPTIONS)
            inputs["Resting ECG Results"] = st.selectbox("Resting ECG Results", ECG_OPTIONS)
            inputs["Exercise Induced Angina"] = st.selectbox("Exercise Induced Angina", YES_NO)
            inputs["Slope of ST Segment"] = st.selectbox("Slope of ST Segment", SLOPE_OPTIONS)
            inputs["Thallium Stress Test Result (thal)"] = st.selectbox("Thallium Stress Test Result (thal)", THAL_OPTIONS)
            inputs["Smoking Status"] = st.selectbox("Smoking Status", SMOKING_OPTIONS)
            inputs["Angeo person or not"] = st.selectbox("Angiography Person (Yes/No)", YES_NO)

        if st.button("Predict"):
            # Convert human inputs to whatever the model expects
            model_inputs = prepare_for_model(inputs, meta)
            df = pd.DataFrame([model_inputs])[num_cols + cat_cols]

            prob = float(model.predict_proba(df)[0, 1])
            label = int(prob >= threshold)

            st.success(f"Predicted probability: **{prob*100:.1f}%**  |  Label (@{threshold:.2f}): **{label}**")

            # Groq or rule-based suggestions
            sug = generate_suggestions(inputs, prob)

            st.markdown("### Summary")
            st.write(sug["summary"])

            colA, colB = st.columns(2)
            with colA:
                st.markdown("#### Immediate Actions")
                for item in sug.get("immediate_actions", []):
                    st.write(f"- {item}")

            with colB:
                st.markdown("#### Lifestyle Suggestions")
                for item in sug.get("lifestyle", []):
                    st.write(f"- {item}")

            st.markdown("#### Additional Recommendations")
            for item in sug.get("recommendations", []):
                st.write(f"- {item}")

            st.markdown(f"**Risk band:** {sug['risk_band']}  |  **Prediction value:** {sug['prediction_value']}")

            # Download text report
            report = StringIO()
            report.write("# Heart Disease Prediction Report\n\n")
            report.write(f"**Risk Probability:** {prob*100:.1f}% (threshold {threshold:.2f})\n\n")
            report.write("## Inputs (human-readable)\n")
            for k, v in inputs.items():
                report.write(f"- {k}: {v}\n")
            report.write("\n## Summary\n")
            report.write(sug["summary"] + "\n\n")
            report.write("## Immediate Actions\n")
            for it in sug.get("immediate_actions", []):
                report.write(f"- {it}\n")
            report.write("\n## Lifestyle Suggestions\n")
            for it in sug.get("lifestyle", []):
                report.write(f"- {it}\n")
            report.write("\n## Additional Recommendations\n")
            for it in sug.get("recommendations", []):
                report.write(f"- {it}\n")
            st.download_button("Download Report (.txt)", report.getvalue(), file_name="heart_report.txt")

    # CSV upload (expects your training column names)
    with tab2:
        st.caption("Upload CSV with the same column names as your training data.")
        upl = st.file_uploader("CSV", type=["csv"])
        if upl is not None:
            df_raw = pd.read_csv(upl)
            missing = [c for c in (num_cols + cat_cols) if c not in df_raw.columns]
            if missing:
                st.error(f"CSV missing required columns: {missing}")
            else:
                probs = model.predict_proba(df_raw[num_cols + cat_cols])[:, 1]
                preds = (probs >= threshold).astype(int)
                out = df_raw.copy()
                out["prediction_proba"] = probs
                out["prediction_label"] = preds
                st.dataframe(out.head(50))
                st.download_button("Download Predictions (.csv)", out.to_csv(index=False), "predictions.csv")

    st.divider()
    st.subheader("Model Info")
    c1, c2 = st.columns(2)
    with c1:
        st.json(meta.get("metrics", {}))
    with c2:
        feature_importance_plot(meta)

if __name__ == "__main__":
    main()
