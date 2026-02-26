"""
model_training.py
=================
Trains and compares multiple ML models for fraud detection.
Saves the best model, scaler, and evaluation artifacts.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import joblib
import warnings

warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models(class_weight_ratio: float = 1.0):
    """Return dict of model name → model instance."""
    # Compute sample weight ratio for imbalanced data
    w = max(class_weight_ratio, 1.0)
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=w,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        ),
    }


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    """
    Train all models, evaluate, pick best by ROC-AUC.
    Returns (results_df, best_model_name, best_model, all_models, eval_artifacts).
    """
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    weight_ratio = neg / max(pos, 1)
    models = get_models(weight_ratio)

    results = []
    all_models = {}
    eval_artifacts = {}

    for name, model in models.items():
        print(f"\n🏋️ Training {name} …")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0

        # Cross-validation (5-fold)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
            "CV F1 Mean": round(cv_scores.mean(), 4),
            "CV F1 Std": round(cv_scores.std(), 4),
        })

        all_models[name] = model
        eval_artifacts[name] = {
            "y_pred": y_pred,
            "y_proba": y_proba,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }

        print(f"   Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

    # ── Isolation Forest (unsupervised baseline) ─────────────────────────
    print("\n🏋️ Training Isolation Forest (unsupervised) …")
    iso = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
    iso.fit(X_train)
    iso_pred = iso.predict(X_test)
    iso_pred_binary = np.where(iso_pred == -1, 1, 0)  # -1 = anomaly = fraud

    iso_acc = accuracy_score(y_test, iso_pred_binary)
    iso_prec = precision_score(y_test, iso_pred_binary, zero_division=0)
    iso_rec = recall_score(y_test, iso_pred_binary, zero_division=0)
    iso_f1 = f1_score(y_test, iso_pred_binary, zero_division=0)

    results.append({
        "Model": "Isolation Forest",
        "Accuracy": round(iso_acc, 4),
        "Precision": round(iso_prec, 4),
        "Recall": round(iso_rec, 4),
        "F1-Score": round(iso_f1, 4),
        "ROC-AUC": "N/A",
        "CV F1 Mean": "N/A",
        "CV F1 Std": "N/A",
    })
    all_models["Isolation Forest"] = iso
    eval_artifacts["Isolation Forest"] = {
        "y_pred": iso_pred_binary,
        "y_proba": None,
        "confusion_matrix": confusion_matrix(y_test, iso_pred_binary),
        "classification_report": classification_report(y_test, iso_pred_binary, output_dict=True),
    }
    print(f"   Acc={iso_acc:.4f}  Prec={iso_prec:.4f}  Rec={iso_rec:.4f}  F1={iso_f1:.4f}")

    # ── Pick best model (by ROC-AUC, excluding Isolation Forest) ─────────
    results_df = pd.DataFrame(results)
    supervised = results_df[results_df["ROC-AUC"] != "N/A"].copy()
    supervised["ROC-AUC"] = supervised["ROC-AUC"].astype(float)
    best_idx = supervised["ROC-AUC"].idxmax()
    best_name = supervised.loc[best_idx, "Model"]

    print(f"\n🏆 Best model: {best_name} (ROC-AUC = {supervised.loc[best_idx, 'ROC-AUC']:.4f})")

    return results_df, best_name, all_models[best_name], all_models, eval_artifacts


def save_model(model, feature_names, best_name, results_df):
    """Save the best model and metadata."""
    model_path = os.path.join(PROJECT_DIR, "model.pkl")
    meta_path = os.path.join(PROJECT_DIR, "model_meta.json")

    joblib.dump(model, model_path)
    print(f"💾 Best model saved to {model_path}")

    meta = {
        "best_model": best_name,
        "feature_names": feature_names,
        "comparison": results_df.to_dict(orient="records"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"💾 Metadata saved to {meta_path}")


def save_evaluation_plots(y_test, eval_artifacts, best_name, feature_names, best_model):
    """Generate and save evaluation plots to disk for the Streamlit app."""
    plots_dir = os.path.join(PROJECT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    best_art = eval_artifacts[best_name]

    # ── Confusion Matrix ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(best_art["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # ── ROC Curve (all supervised models) ────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for mname, art in eval_artifacts.items():
        if art["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, art["y_proba"])
            row = [r for r in eval_artifacts[mname]["classification_report"].values()
                   if isinstance(r, dict)]
            auc_val = roc_auc_score(y_test, art["y_proba"])
            ax.plot(fpr, tpr, label=f"{mname} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "roc_curve.png"), dpi=150)
    plt.close(fig)

    # ── Precision-Recall Curve ───────────────────────────────────────────
    if best_art["y_proba"] is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        prec_vals, rec_vals, _ = precision_recall_curve(y_test, best_art["y_proba"])
        ax.plot(rec_vals, prec_vals, color="darkorange")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve — {best_name}")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "precision_recall.png"), dpi=150)
        plt.close(fig)

    # ── Feature Importance ───────────────────────────────────────────────
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": imp})
        fi_df.sort_values("Importance", ascending=True, inplace=True)
        fi_df = fi_df.tail(20)  # top 20

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(fi_df["Feature"], fi_df["Importance"], color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top Feature Importances — {best_name}")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "feature_importance.png"), dpi=150)
        plt.close(fig)

    print(f"📊 Plots saved to {plots_dir}/")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    from data_processing import run_full_pipeline

    X_train, X_test, y_train, y_test, scaler, feature_names, target_col, df_raw = (
        run_full_pipeline()
    )

    results_df, best_name, best_model, all_models, eval_artifacts = train_and_evaluate(
        X_train, X_test, y_train, y_test, feature_names
    )

    print("\n" + "=" * 60)
    print("📋 MODEL COMPARISON TABLE")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)

    save_model(best_model, feature_names, best_name, results_df)
    save_evaluation_plots(y_test, eval_artifacts, best_name, feature_names, best_model)

    print("\n🎉 Training complete! Run `streamlit run app.py` to launch the dashboard.")


if __name__ == "__main__":
    main()
