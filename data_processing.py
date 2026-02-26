"""
data_processing.py
==================
Handles data loading, merging, cleaning, encoding, scaling, and train/test splitting.
Supports both PaySim-format datasets and creditcard.csv fallback.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

PAYSIM_FILES = [
    "PS_20174392719_1491204439457_log.csv",
    "Synthetic_Financial_datasets_log.csv",
]
CREDITCARD_FILE = "creditcard.csv"

# Columns to drop (not useful for modelling)
DROP_COLS = ["nameOrig", "nameDest", "isFlaggedFraud"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_merge_data() -> pd.DataFrame:
    """Load and merge all available PaySim CSVs. Falls back to creditcard.csv."""
    frames = []
    for fname in PAYSIM_FILES:
        fpath = os.path.join(PROJECT_DIR, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            df["_source"] = fname
            frames.append(df)

    if frames:
        data = pd.concat(frames, ignore_index=True)
        print(f"✅ Loaded {len(data)} rows from {len(frames)} PaySim file(s)")
        return data

    # Fallback: creditcard.csv
    cc_path = os.path.join(PROJECT_DIR, CREDITCARD_FILE)
    if os.path.exists(cc_path):
        data = pd.read_csv(cc_path)
        print(f"✅ Loaded {len(data)} rows from creditcard.csv (fallback)")
        return data

    raise FileNotFoundError("No dataset found in the project directory.")


def detect_target_column(df: pd.DataFrame) -> str:
    """Auto-detect the target column name."""
    candidates = ["isFraud", "is_fraud", "Class", "class", "fraud", "Fraud", "label", "Label"]
    for col in candidates:
        if col in df.columns:
            print(f"🎯 Target column detected: '{col}'")
            return col
    raise ValueError(f"Cannot auto-detect target column. Columns: {list(df.columns)}")


def clean_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Handle missing values, drop unnecessary columns, encode categoricals."""
    df = df.copy()

    # Drop helper columns
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if "_source" in df.columns:
        cols_to_drop.append("_source")
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # Handle missing values
    num_missing = df.isnull().sum().sum()
    if num_missing > 0:
        print(f"⚠️  Filling {num_missing} missing values")
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col].fillna(df[col].median(), inplace=True)
        for col in df.select_dtypes(include=["object", "category"]).columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        print("✅ No missing values")

    # Encode categorical columns (e.g., 'type')
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    if cat_cols:
        print(f"🔄 One-hot encoding: {cat_cols}")
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)

    return df


def split_and_scale(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    apply_smote: bool = True,
):
    """
    Train/test split → scale features → optionally apply SMOTE.
    Returns (X_train, X_test, y_train, y_test, scaler, feature_names).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    feature_names = list(X.columns)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"📊 Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"📊 Fraud rate — Train: {y_train.mean():.2%} | Test: {y_test.mean():.2%}")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )

    # SMOTE for class imbalance (only on training data)
    if apply_smote:
        fraud_count = y_train.sum()
        if fraud_count < 6:
            # Too few fraud samples for SMOTE — use simple duplication
            print("⚠️  Too few fraud samples for SMOTE, using class weights instead")
        else:
            k = min(5, fraud_count - 1)
            smote = SMOTE(random_state=42, k_neighbors=k)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            print(f"⚖️  After SMOTE — Train size: {len(X_train_scaled)}, Fraud: {y_train.sum()}")

    # Save scaler
    scaler_path = os.path.join(PROJECT_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"💾 Scaler saved to {scaler_path}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


# ---------------------------------------------------------------------------
# Convenience: full pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline():
    """Run the complete data processing pipeline. Returns processed data."""
    df_raw = load_and_merge_data()
    target_col = detect_target_column(df_raw)

    # We import feature engineering here to avoid circular imports
    from feature_engineering import engineer_features

    df_featured = engineer_features(df_raw, target_col)
    df_clean = clean_data(df_featured, target_col)
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(
        df_clean, target_col
    )
    return X_train, X_test, y_train, y_test, scaler, feature_names, target_col, df_raw


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, feature_names, target, raw = run_full_pipeline()
    print(f"\n✅ Pipeline complete! Features: {len(feature_names)}")
    print(f"Feature names: {feature_names}")
