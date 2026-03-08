"""
Stock Market Signal Classification
-----------------------------------
Trains Logistic Regression, Random Forest,
and XGBoost to classify NSE stock signals
as Bullish or Bearish using technical indicators.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE    = "/home/deepak/stock_pipeline"
VIZ     = f"{BASE}/data/viz"
MODELS  = f"{BASE}/models"
REPORTS = f"{BASE}/data/reports"

os.makedirs(MODELS,  exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)

FEATURES = [
    "Open", "High", "Low", "Close",
    "Volume", "Daily_Return",
    "Volatility_7", "RSI", "Month",
]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load processed stock features from CSV."""
    print("Loading processed data ...")
    df = pd.read_csv(f"{VIZ}/stock_features.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    print(f"  Records : {len(df):,}")
    print(f"  Stocks  : {df['Symbol'].nunique()}")
    return df


# ---------------------------------------------------------------------------
# Target Engineering
# ---------------------------------------------------------------------------
def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use MA crossover signal (Bullish / Bearish) as classification target.
    Neutral class is dropped due to insufficient samples (~50 records).
    """
    print("\nCreating target labels ...")
    df = df.copy()
    df["Target"] = df["MA_Signal"]
    df = df[df["Target"] != "Neutral"]
    df = df.dropna(subset=["Target"])

    dist = df["Target"].value_counts()
    pct  = df["Target"].value_counts(normalize=True).mul(100).round(1)
    print(f"  Distribution:\n{dist.to_string()}")
    print(f"\n  Percentages:\n{pct.to_string()}")
    return df


# ---------------------------------------------------------------------------
# Feature Preparation
# ---------------------------------------------------------------------------
def prepare_features(df: pd.DataFrame):
    """Scale features and encode target labels."""
    print("\nPreparing features ...")

    X = df[FEATURES].copy().fillna(df[FEATURES].median())
    y = df["Target"].copy()

    le        = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, f"{MODELS}/scaler.pkl")
    joblib.dump(le,     f"{MODELS}/label_encoder.pkl")

    print(f"  Features : {FEATURES}")
    print(f"  Classes  : {list(le.classes_)}")
    print(f"  X shape  : {X.shape}")
    return X_scaled, y_encoded, le, X


# ---------------------------------------------------------------------------
# Train / Test Split
# ---------------------------------------------------------------------------
def split_data(X, y):
    """Stratified 80/20 split."""
    print("\nSplitting data ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train : {len(X_train):,} samples")
    print(f"  Test  : {len(X_test):,} samples")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------
def train_models(X_train, X_test, y_train, y_test, le) -> dict:
    """Train Logistic Regression, Random Forest, and XGBoost."""
    print("\nTraining models ...")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10,
            random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=6,
            learning_rate=0.1, random_state=42,
            eval_metric="mlogloss", verbosity=0
        ),
    }

    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        print(f"\n  [{name}]")
        model.fit(X_train, y_train)
        y_pred    = model.predict(X_test)
        acc       = accuracy_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred, average="weighted")
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="accuracy"
        )
        results[name] = {
            "model":    model,
            "accuracy": acc,
            "f1_score": f1,
            "cv_mean":  cv_scores.mean(),
            "cv_std":   cv_scores.std(),
            "y_pred":   y_pred,
        }
        print(f"    Accuracy : {acc*100:.2f}%")
        print(f"    F1 Score : {f1:.4f}")
        print(f"    CV Score : {cv_scores.mean()*100:.2f}% "
              f"(+/- {cv_scores.std()*100:.2f}%)")

    return results


# ---------------------------------------------------------------------------
# Model Selection
# ---------------------------------------------------------------------------
def select_best_model(results: dict, le) -> tuple:
    """Print comparison table and return best model by accuracy."""
    print("\nModel Comparison")
    print("=" * 56)
    print(f"  {'Model':<22} {'Accuracy':>10} {'F1':>8} {'CV Mean':>10}")
    print("-" * 56)

    best_name, best_score = None, 0.0
    for name, r in results.items():
        print(f"  {name:<22} {r['accuracy']*100:>9.2f}%"
              f" {r['f1_score']:>8.4f} {r['cv_mean']*100:>9.2f}%")
        if r["accuracy"] > best_score:
            best_score = r["accuracy"]
            best_name  = name

    print("=" * 56)
    print(f"  Best Model : {best_name} ({best_score*100:.2f}%)")
    return best_name, results[best_name]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_best_model(best_name, best_result, y_test, le) -> None:
    """Print classification report and save confusion matrix."""
    print(f"\nDetailed Evaluation — {best_name}")
    print("=" * 56)
    print(classification_report(
        y_test, best_result["y_pred"], target_names=le.classes_
    ))

    cm = confusion_matrix(y_test, best_result["y_pred"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=le.classes_, yticklabels=le.classes_,
    )
    plt.title(f"Confusion Matrix — {best_name}",
              fontsize=14, fontweight="bold")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{REPORTS}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {REPORTS}/confusion_matrix.png")


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------
def plot_feature_importance(best_name, best_result) -> None:
    """Bar chart of feature importances (or LR coefficients)."""
    print(f"\nFeature Importance — {best_name}")
    model = best_result["model"]

    importance = (
        np.abs(model.coef_).mean(axis=0)
        if best_name == "Logistic Regression"
        else model.feature_importances_
    )

    feat_df = (
        pd.DataFrame({"Feature": FEATURES, "Importance": importance})
        .sort_values("Importance")
    )

    plt.figure(figsize=(10, 6))
    plt.barh(feat_df["Feature"], feat_df["Importance"],
             color="steelblue", edgecolor="white")
    plt.title(f"Feature Importance — {best_name}",
              fontsize=14, fontweight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(f"{REPORTS}/feature_importance.png", dpi=150)
    plt.close()
    print(f"  Plot saved to {REPORTS}/feature_importance.png")

    top5 = feat_df.sort_values("Importance", ascending=False).head(5)
    print(f"\n  Top 5 Features:\n{top5.to_string(index=False)}")


# ---------------------------------------------------------------------------
# SHAP Explainability
# ---------------------------------------------------------------------------
def shap_explainability(best_name, best_result, X_test) -> None:
    """Generate and save SHAP summary plot."""
    print(f"\nSHAP Explainability — {best_name}")
    model    = best_result["model"]
    X_sample = X_test[:200]

    try:
        explainer = (
            shap.TreeExplainer(model)
            if best_name in ("XGBoost", "Random Forest")
            else shap.LinearExplainer(model, X_test[:100])
        )
        shap_values = explainer.shap_values(X_sample)

        sv = shap_values[0] if isinstance(shap_values, list) \
             else shap_values

        plt.figure()
        shap.summary_plot(
            sv, X_sample, feature_names=FEATURES,
            show=False, plot_size=(10, 6)
        )
        plt.title("SHAP Feature Impact",
                  fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{REPORTS}/shap_summary.png", dpi=150)
        plt.close()
        print(f"  SHAP plot saved to {REPORTS}/shap_summary.png")

    except Exception as exc:
        print(f"  SHAP skipped — {exc}")


# ---------------------------------------------------------------------------
# Persist Model
# ---------------------------------------------------------------------------
def save_model(best_name, best_result) -> None:
    """Serialize the best model and its name to disk."""
    print("\nSaving best model ...")
    joblib.dump(best_result["model"], f"{MODELS}/best_model.pkl")
    joblib.dump(best_name,            f"{MODELS}/best_model_name.pkl")
    print(f"  Model : {best_name}")
    print(f"  Path  : {MODELS}/best_model.pkl")


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
def save_predictions(df, best_result, le, X_raw) -> pd.DataFrame:
    """Run inference on full dataset and persist results."""
    print("\nSaving predictions ...")
    scaler      = joblib.load(f"{MODELS}/scaler.pkl")
    X_all       = scaler.transform(X_raw)
    pred_labels = le.inverse_transform(best_result["model"].predict(X_all))

    df = df.copy()
    df["Predicted_Signal"] = pred_labels
    df["Correct"]          = df["Target"] == df["Predicted_Signal"]
    df.to_csv(f"{VIZ}/predictions.csv", index=False)

    print(f"  Records saved : {len(df):,}")
    print(f"  Path          : {VIZ}/predictions.csv")

    acc_by_stock = (
        df.groupby("Symbol")["Correct"]
          .mean().mul(100).round(2)
          .rename("Accuracy_%")
    )
    print(f"\n  Accuracy by Stock:\n{acc_by_stock.to_string()}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Stock Market Signal Classification")
    print("=" * 56)

    df                              = load_data()
    df                              = create_target(df)
    X_scaled, y_enc, le, X_raw     = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y_enc)
    results                         = train_models(
                                          X_train, X_test,
                                          y_train, y_test, le
                                      )
    best_name, best_result          = select_best_model(results, le)

    evaluate_best_model(best_name, best_result, y_test, le)
    plot_feature_importance(best_name, best_result)
    shap_explainability(best_name, best_result, X_test)
    save_model(best_name, best_result)
    save_predictions(df, best_result, le, X_raw)

    print("\n" + "=" * 56)
    print("Output Files")
    print("-" * 56)
    print(f"  {MODELS}/best_model.pkl")
    print(f"  {MODELS}/scaler.pkl")
    print(f"  {MODELS}/label_encoder.pkl")
    print(f"  {REPORTS}/confusion_matrix.png")
    print(f"  {REPORTS}/feature_importance.png")
    print(f"  {REPORTS}/shap_summary.png")
    print(f"  {VIZ}/predictions.csv")
    print("=" * 56)
