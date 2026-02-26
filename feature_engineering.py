"""
feature_engineering.py
======================
Engineers behavioral features from raw transaction data.
Focuses on anomaly signals: balance discrepancies, velocity, amount patterns.
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def engineer_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Add behavioural / anomaly features to the raw dataframe.
    Works with PaySim-schema data. If key columns are missing, returns
    the dataframe unchanged (e.g. creditcard.csv already has PCA features).
    """
    df = df.copy()

    # ── Check if this is PaySim-format data ──────────────────────────────
    paysim_cols = {"amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"}
    if not paysim_cols.issubset(set(df.columns)):
        print("ℹ️  Non-PaySim dataset detected — skipping custom feature engineering")
        return df

    print("🔧 Engineering behavioral features …")

    # ── 1. Balance-change ratio (sender) ─────────────────────────────────
    df["balance_change_ratio"] = np.where(
        df["oldbalanceOrg"] > 0,
        (df["oldbalanceOrg"] - df["newbalanceOrig"]) / df["oldbalanceOrg"],
        0.0,
    )

    # ── 2. Amount-to-balance ratio ───────────────────────────────────────
    df["amount_to_balance_ratio"] = np.where(
        df["oldbalanceOrg"] > 0,
        df["amount"] / df["oldbalanceOrg"],
        0.0,
    )

    # ── 3. Balance error (expected vs actual) ────────────────────────────
    #   If the books don't balance, that's suspicious.
    df["balance_error_orig"] = (
        df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    )
    df["balance_error_dest"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    )

    # ── 4. Is-merchant flag ──────────────────────────────────────────────
    if "nameDest" in df.columns:
        df["is_merchant"] = df["nameDest"].str.startswith("M").astype(int)

    # ── 5. High-risk transaction type ────────────────────────────────────
    if "type" in df.columns:
        df["is_high_risk_type"] = df["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int)

    # ── 6. Sender-level aggregate stats (velocity / frequency) ───────────
    if "nameOrig" in df.columns:
        sender_stats = df.groupby("nameOrig")["amount"].agg(
            sender_txn_count="count",
            sender_avg_amount="mean",
            sender_std_amount="std",
            sender_max_amount="max",
        ).reset_index()
        sender_stats["sender_std_amount"].fillna(0, inplace=True)
        df = df.merge(sender_stats, on="nameOrig", how="left")

        # Z-score of amount relative to sender's own history
        df["amount_zscore"] = np.where(
            df["sender_std_amount"] > 0,
            (df["amount"] - df["sender_avg_amount"]) / df["sender_std_amount"],
            0.0,
        )

    # ── 7. Time-based features (step = simulated hour) ───────────────────
    if "step" in df.columns:
        df["hour_of_day"] = df["step"] % 24
        df["is_night"] = df["hour_of_day"].isin(range(0, 6)).astype(int)
        df["day_number"] = df["step"] // 24

    # ── 8. Large-amount flag ─────────────────────────────────────────────
    amount_95 = df["amount"].quantile(0.95)
    df["is_large_amount"] = (df["amount"] >= amount_95).astype(int)

    # ── 9. Empty-account-after-txn flag ──────────────────────────────────
    df["emptied_account"] = (df["newbalanceOrig"] == 0).astype(int) & (df["oldbalanceOrg"] > 0).astype(int)

    n_new = len([c for c in df.columns if c not in paysim_cols and c != target_col])
    print(f"✅ Engineered {n_new} total features (including one-hot later)")

    return df


def get_feature_descriptions() -> dict:
    """Return human-readable descriptions for engineered features."""
    return {
        "balance_change_ratio": "Fraction of sender balance consumed by transaction",
        "amount_to_balance_ratio": "Transaction amount relative to sender's balance",
        "balance_error_orig": "Discrepancy in sender's expected vs actual balance",
        "balance_error_dest": "Discrepancy in receiver's expected vs actual balance",
        "is_merchant": "Whether the destination is a merchant account",
        "is_high_risk_type": "Whether the transaction is TRANSFER or CASH_OUT",
        "sender_txn_count": "Number of transactions by this sender",
        "sender_avg_amount": "Average transaction amount for this sender",
        "sender_std_amount": "Std deviation of transaction amounts for this sender",
        "sender_max_amount": "Max transaction amount for this sender",
        "amount_zscore": "Z-score of amount vs sender's transaction history",
        "hour_of_day": "Hour of day (0-23) from simulation step",
        "is_night": "Whether transaction occurred at night (0-5 AM)",
        "day_number": "Day number derived from simulation step",
        "is_large_amount": "Whether amount is in the top 5%",
        "emptied_account": "Whether sender's balance was emptied to zero",
    }
