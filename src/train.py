import argparse
import json
import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_NUMERIC = [
    "Age",
    "Resting Blood Pressure",
    "Total Cholesterol",
    "Fasting Blood Sugar",
    "Maximum Heart Rate Achieved",
    "ST Depression (oldpeak)",
    "Number of Major Vessels",
    "Body Mass Index (BMI)"
]

DEFAULT_CATEGORICAL = [
    "Sex",
    "Chest Pain Type",
    "Resting ECG Results",
    "Exercise Induced Angina",
    "Slope of ST Segment",
    "Thallium Stress Test Result (thal)",
    "Smoking Status",
    "Angeo person or not"
]

TARGET_COL = "Heart disease"

def infer_columns(df):
    """Ensure required columns exist and split into numeric/categorical robustly."""
    missing = [c for c in [*DEFAULT_NUMERIC, *DEFAULT_CATEGORICAL, TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}\nFound: {list(df.columns)}")
    # Use defaults but keep only those present (robust)
    num_cols = [c for c in DEFAULT_NUMERIC if c in df.columns]
    cat_cols = [c for c in DEFAULT_CATEGORICAL if c in df.columns]
    return num_cols, cat_cols

def build_pipeline(num_cols, cat_cols):
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )
    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    pipe = Pipeline(steps=[
        ("preprocessor", pre),
        ("model", rf)
    ])
    return pipe

def get_ohe_feature_names(preprocessor, cat_cols, num_cols):
    """Return final feature names after ColumnTransformer for RF feature_importances_."""
    num_features = list(num_cols)
    ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
    ohe_features = list(ohe.get_feature_names_out(cat_cols))
    return num_features + ohe_features

def main(args):
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    df = pd.read_csv(args.data)
    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COL]).copy()

    num_cols, cat_cols = infer_columns(df)

    # Cast categoricals to string to be safe (OHE expects object/string)
    for c in cat_cols:
        df[c] = df[c].astype(str)

    X = df[num_cols + cat_cols]
    y = df[TARGET_COL].astype(int)  # ensure 0/1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    pipe = build_pipeline(num_cols, cat_cols)

    # Hyperparameter search (balanced for precision & generalization)
    param_dist = {
        "model__n_estimators": [300, 400, 600, 800],
        "model__max_depth": [None, 8, 12, 16, 24],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 3, 4],
        "model__max_features": ["sqrt", "log2", 0.5],
        "model__bootstrap": [True, False],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=30,
        scoring="average_precision",  # emphasizes precision across recall levels
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    search.fit(X_train, y_train)

    best = search.best_estimator_
    print("\nBest params:", search.best_params_)

    # Evaluate
    y_proba = best.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "average_precision": float(average_precision_score(y_test, y_proba)),
        "precision_at_0_5": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_test, y_pred, zero_division=0))
    }

    print("\nClassification report (@0.5 threshold):\n",
          classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC AUC:", metrics["roc_auc"])
    print("Average Precision (AP):", metrics["average_precision"])
    print("Precision@0.5:", metrics["precision_at_0_5"], "Recall@0.5:", metrics["recall_at_0_5"])

    # Persist model
    joblib.dump(best, args.model_out)

    # Build feature importance mapping
    pre = best.named_steps["preprocessor"]
    feature_names = get_ohe_feature_names(pre, cat_cols, num_cols)
    rf = best.named_steps["model"]
    importances = rf.feature_importances_
    fi = sorted(
        [{"feature": f, "importance": float(i)} for f, i in zip(feature_names, importances)],
        key=lambda d: d["importance"], reverse=True
    )

    # Save metadata (incl. categories for UI dropdowns)
    metadata = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "target": TARGET_COL,
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "categories": {c: sorted([str(v) for v in df[c].dropna().unique().tolist()]) for c in cat_cols},
        "metrics": metrics,
        "feature_importances": fi
    }
    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model → {args.model_out}")
    print(f"Saved metadata → {args.meta_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/heart.csv")
    parser.add_argument("--model_out", type=str, default="models/model.joblib")
    parser.add_argument("--meta_out", type=str, default="models/metadata.json")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()
    main(args)
