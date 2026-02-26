"""
app.py
======
Streamlit-based Fraud Detection Dashboard.
Pages: Dashboard · Real-Time Prediction · Model Insights
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config & paths
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

MODEL_PATH = os.path.join(PROJECT_DIR, "model.pkl")
SCALER_PATH = os.path.join(PROJECT_DIR, "scaler.pkl")
META_PATH = os.path.join(PROJECT_DIR, "model_meta.json")
PLOTS_DIR = os.path.join(PROJECT_DIR, "plots")

st.set_page_config(
    page_title="🛡️ Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for premium look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 24px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25);
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-4px); }
    .kpi-card h2 { font-size: 2.2rem; margin: 0; font-weight: 700; }
    .kpi-card p { font-size: 0.9rem; margin: 4px 0 0 0; opacity: 0.85; }

    .kpi-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        box-shadow: 0 8px 32px rgba(17, 153, 142, 0.25);
    }
    .kpi-red {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        box-shadow: 0 8px 32px rgba(235, 51, 73, 0.25);
    }
    .kpi-blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        box-shadow: 0 8px 32px rgba(79, 172, 254, 0.25);
    }
    .kpi-orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 8px 32px rgba(240, 147, 251, 0.25);
    }

    /* Risk gauge */
    .risk-low { color: #38ef7d; font-weight: 700; }
    .risk-medium { color: #f5a623; font-weight: 700; }
    .risk-high { color: #eb3349; font-weight: 700; }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 1.8rem;
        margin-bottom: 1rem;
    }

    /* Prediction result box */
    .pred-box {
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 16px;
    }
    .pred-legit {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        color: #1a5c1a;
    }
    .pred-fraud {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_raw_data():
    from data_processing import load_and_merge_data, detect_target_column
    df = load_and_merge_data()
    target = detect_target_column(df)
    return df, target


@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    meta = {}
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
    return model, scaler, meta


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🛡️ Fraud Detection")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "🔮 Real-Time Prediction", "🧠 Model Insights"],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.caption("Behavioral Fraud Detection System v1.0")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 : DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown('<p class="section-header">📊 Dashboard Overview</p>', unsafe_allow_html=True)

    df_raw, target_col = load_raw_data()
    model, scaler, meta = load_model_artifacts()

    total_txn = len(df_raw)
    fraud_count = int(df_raw[target_col].sum())
    legit_count = total_txn - fraud_count
    fraud_rate = fraud_count / total_txn if total_txn > 0 else 0

    # Model accuracy from metadata
    model_acc = "N/A"
    best_model_name = meta.get("best_model", "N/A")
    if "comparison" in meta:
        for row in meta["comparison"]:
            if row["Model"] == best_model_name:
                model_acc = f"{row.get('Accuracy', 0) * 100:.1f}%"
                break

    # ── KPI cards ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="kpi-card kpi-blue"><h2>{total_txn:,}</h2><p>Total Transactions</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card kpi-green"><h2>{legit_count:,}</h2><p>Legitimate</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card kpi-red"><h2>{fraud_count:,}</h2><p>Fraudulent</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi-card kpi-orange"><h2>{fraud_rate:.2%}</h2><p>Fraud Rate</p></div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Best model info ──────────────────────────────────────────────────
    st.info(f"🏆 **Best Model:** {best_model_name}  |  **Test Accuracy:** {model_acc}")

    # ── Charts ───────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fraud vs Legitimate Distribution")
        fig_pie = px.pie(
            names=["Legitimate", "Fraud"],
            values=[legit_count, fraud_count],
            color_discrete_sequence=["#38ef7d", "#eb3349"],
            hole=0.45,
        )
        fig_pie.update_layout(
            font=dict(family="Inter"), margin=dict(t=30, b=0, l=0, r=0),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.3),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Transaction Amount Distribution")
        if "amount" in df_raw.columns:
            fig_hist = px.histogram(
                df_raw, x="amount", color=df_raw[target_col].map({0: "Legit", 1: "Fraud"}),
                nbins=50, color_discrete_map={"Legit": "#4facfe", "Fraud": "#eb3349"},
                barmode="overlay", opacity=0.7,
            )
            fig_hist.update_layout(
                font=dict(family="Inter"), xaxis_title="Amount", yaxis_title="Count",
                legend_title="", margin=dict(t=30, b=0),
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.write("Amount column not available for this dataset.")

    # ── Time-based fraud pattern ─────────────────────────────────────────
    if "step" in df_raw.columns:
        st.subheader("Fraud Transactions Over Time (by Simulation Step)")
        fraud_by_step = df_raw[df_raw[target_col] == 1].groupby("step").size().reset_index(name="Fraud Count")
        fig_time = px.line(
            fraud_by_step, x="step", y="Fraud Count",
            color_discrete_sequence=["#eb3349"],
        )
        fig_time.update_layout(
            font=dict(family="Inter"), xaxis_title="Time Step",
            margin=dict(t=30, b=0),
        )
        st.plotly_chart(fig_time, use_container_width=True)

    # ── Transaction type breakdown ───────────────────────────────────────
    if "type" in df_raw.columns:
        st.subheader("Fraud by Transaction Type")
        type_fraud = df_raw.groupby("type")[target_col].agg(["sum", "count"]).reset_index()
        type_fraud.columns = ["Type", "Fraud", "Total"]
        type_fraud["Fraud Rate"] = type_fraud["Fraud"] / type_fraud["Total"]
        fig_bar = px.bar(
            type_fraud, x="Type", y="Fraud Rate",
            color="Fraud Rate", color_continuous_scale="Reds",
            text=type_fraud["Fraud Rate"].apply(lambda x: f"{x:.1%}"),
        )
        fig_bar.update_layout(font=dict(family="Inter"), margin=dict(t=30, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 : REAL-TIME PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔮 Real-Time Prediction":
    st.markdown('<p class="section-header">🔮 Real-Time Fraud Prediction</p>', unsafe_allow_html=True)

    model, scaler, meta = load_model_artifacts()
    feature_names = meta.get("feature_names", [])

    if model is None or scaler is None:
        st.error("⚠️ Model not found! Please run `python model_training.py` first.")
        st.stop()

    df_raw, target_col = load_raw_data()

    # Detect whether this is PaySim data
    is_paysim = "type" in df_raw.columns

    st.markdown("Enter transaction details below to get a fraud prediction.")

    with st.form("prediction_form"):
        if is_paysim:
            col1, col2 = st.columns(2)
            with col1:
                txn_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
                amount = st.number_input("Amount ($)", min_value=0.0, value=1000.0, step=100.0)
                old_balance_org = st.number_input("Sender's Old Balance ($)", min_value=0.0, value=10000.0, step=500.0)
            with col2:
                new_balance_org = st.number_input("Sender's New Balance ($)", min_value=0.0, value=9000.0, step=500.0)
                old_balance_dest = st.number_input("Receiver's Old Balance ($)", min_value=0.0, value=5000.0, step=500.0)
                new_balance_dest = st.number_input("Receiver's New Balance ($)", min_value=0.0, value=6000.0, step=500.0)

            step = st.slider("Time Step (simulation hour)", 1, 744, 100)
            is_merchant_input = st.checkbox("Destination is a Merchant", value=False)

            submitted = st.form_submit_button("🔍 Predict Fraud", use_container_width=True)
        else:
            st.info("Enter feature values for prediction (creditcard.csv format).")
            inputs = {}
            cols = st.columns(4)
            for i, feat in enumerate(feature_names):
                with cols[i % 4]:
                    inputs[feat] = st.number_input(feat, value=0.0, format="%.6f", key=feat)
            submitted = st.form_submit_button("🔍 Predict Fraud", use_container_width=True)

    if submitted:
        try:
            if is_paysim:
                # Build raw feature dict matching engineered features
                raw_input = {
                    "step": step,
                    "amount": amount,
                    "oldbalanceOrg": old_balance_org,
                    "newbalanceOrig": new_balance_org,
                    "oldbalanceDest": old_balance_dest,
                    "newbalanceDest": new_balance_dest,
                }

                # Engineered features
                raw_input["balance_change_ratio"] = (
                    (old_balance_org - new_balance_org) / old_balance_org if old_balance_org > 0 else 0.0
                )
                raw_input["amount_to_balance_ratio"] = (
                    amount / old_balance_org if old_balance_org > 0 else 0.0
                )
                raw_input["balance_error_orig"] = old_balance_org - amount - new_balance_org
                raw_input["balance_error_dest"] = old_balance_dest + amount - new_balance_dest
                raw_input["is_merchant"] = int(is_merchant_input)
                raw_input["is_high_risk_type"] = int(txn_type in ["TRANSFER", "CASH_OUT"])

                # Sender aggregate stats (use dataset median as proxy for single txn)
                raw_input["sender_txn_count"] = 1
                raw_input["sender_avg_amount"] = amount
                raw_input["sender_std_amount"] = 0.0
                raw_input["sender_max_amount"] = amount
                raw_input["amount_zscore"] = 0.0

                # Time features
                raw_input["hour_of_day"] = step % 24
                raw_input["is_night"] = int(raw_input["hour_of_day"] in range(0, 6))
                raw_input["day_number"] = step // 24

                # Derived flags
                amount_95 = df_raw["amount"].quantile(0.95)
                raw_input["is_large_amount"] = int(amount >= amount_95)
                raw_input["emptied_account"] = int(new_balance_org == 0 and old_balance_org > 0)

                # One-hot encode transaction type
                type_categories = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
                for cat in type_categories:
                    raw_input[f"type_{cat}"] = int(txn_type == cat)

                # Build DataFrame with correct feature order
                input_df = pd.DataFrame([raw_input])
                # Keep only the features the model expects
                for feat in feature_names:
                    if feat not in input_df.columns:
                        input_df[feat] = 0.0
                input_df = input_df[feature_names]

            else:
                input_df = pd.DataFrame([inputs])[feature_names]

            # Scale and predict
            input_scaled = scaler.transform(input_df)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_scaled)[0][1]
                prediction = int(proba >= 0.5)
            else:
                raw_pred = model.predict(input_scaled)[0]
                prediction = int(raw_pred == -1) if hasattr(model, "offset_") else int(raw_pred)
                proba = float(prediction)

            # ── Display result ───────────────────────────────────────────
            st.markdown("---")
            risk_pct = proba * 100

            if prediction == 0:
                st.markdown(
                    f'<div class="pred-box pred-legit">✅ LEGITIMATE TRANSACTION<br>'
                    f'Risk Score: {risk_pct:.1f}%</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="pred-box pred-fraud">🚨 FRAUDULENT TRANSACTION DETECTED<br>'
                    f'Risk Score: {risk_pct:.1f}%</div>',
                    unsafe_allow_html=True,
                )

            # Risk gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_pct,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Fraud Risk Score", "font": {"size": 20}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#eb3349" if risk_pct > 50 else "#38ef7d"},
                    "steps": [
                        {"range": [0, 30], "color": "#d4fc79"},
                        {"range": [30, 70], "color": "#f5a623"},
                        {"range": [70, 100], "color": "#ff416c"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": risk_pct,
                    },
                },
            ))
            fig_gauge.update_layout(height=300, margin=dict(t=50, b=0, l=40, r=40))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ── SHAP explanation ─────────────────────────────────────────
            if hasattr(model, "predict_proba"):
                st.subheader("🔍 Prediction Explanation (Feature Contributions)")
                try:
                    import shap
                    explainer = shap.TreeExplainer(model) if hasattr(model, "feature_importances_") else shap.LinearExplainer(model, input_scaled)
                    shap_values = explainer.shap_values(input_scaled)

                    # Handle different SHAP output shapes
                    if isinstance(shap_values, list):
                        sv = shap_values[1]  # class 1 (fraud)
                    else:
                        sv = shap_values

                    sv_flat = sv.flatten()
                    shap_df = pd.DataFrame({
                        "Feature": feature_names,
                        "SHAP Value": sv_flat,
                        "Abs SHAP": np.abs(sv_flat),
                    }).sort_values("Abs SHAP", ascending=False).head(15)

                    fig_shap = px.bar(
                        shap_df, x="SHAP Value", y="Feature", orientation="h",
                        color="SHAP Value",
                        color_continuous_scale=["#38ef7d", "#f5a623", "#eb3349"],
                        color_continuous_midpoint=0,
                    )
                    fig_shap.update_layout(
                        yaxis={"categoryorder": "total ascending"},
                        height=400, margin=dict(t=30, b=0),
                        font=dict(family="Inter"),
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
                except Exception as e:
                    st.warning(f"SHAP explanation unavailable: {e}")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.exception(e)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 : MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🧠 Model Insights":
    st.markdown('<p class="section-header">🧠 Model Insights</p>', unsafe_allow_html=True)

    model, scaler, meta = load_model_artifacts()
    best_name = meta.get("best_model", "N/A")
    feature_names = meta.get("feature_names", [])

    # ── Model Comparison Table ───────────────────────────────────────────
    st.subheader("📋 Model Comparison")
    if "comparison" in meta:
        cmp_df = pd.DataFrame(meta["comparison"])
        st.dataframe(cmp_df.style.highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1-Score"],
            color="#d4fc79", axis=0,
        ), use_container_width=True)
    else:
        st.warning("No model comparison data found. Run `python model_training.py` first.")

    st.info(f"🏆 **Selected Best Model:** {best_name}")

    # ── Saved plots ──────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    roc_path = os.path.join(PLOTS_DIR, "roc_curve.png")
    pr_path = os.path.join(PLOTS_DIR, "precision_recall.png")
    fi_path = os.path.join(PLOTS_DIR, "feature_importance.png")

    with col1:
        st.subheader("Confusion Matrix")
        if os.path.exists(cm_path):
            st.image(cm_path)
        else:
            st.warning("Plot not found. Run model_training.py first.")

    with col2:
        st.subheader("ROC Curve")
        if os.path.exists(roc_path):
            st.image(roc_path)
        else:
            st.warning("Plot not found. Run model_training.py first.")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Precision-Recall Curve")
        if os.path.exists(pr_path):
            st.image(pr_path)
        else:
            st.warning("Plot not found. Run model_training.py first.")

    with col4:
        st.subheader("Feature Importance")
        if os.path.exists(fi_path):
            st.image(fi_path)
        else:
            st.warning("Plot not found. Run model_training.py first.")

    # ── Interactive Feature Importance ────────────────────────────────────
    if model is not None and hasattr(model, "feature_importances_"):
        st.subheader("Interactive Feature Importance")
        imp = model.feature_importances_
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": imp})
        fi_df.sort_values("Importance", ascending=True, inplace=True)
        fi_df = fi_df.tail(20)

        fig_fi = px.bar(
            fi_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Viridis",
        )
        fig_fi.update_layout(
            height=500, margin=dict(t=30, b=0),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Feature descriptions ─────────────────────────────────────────────
    st.subheader("📖 Behavioral Feature Descriptions")
    from feature_engineering import get_feature_descriptions
    desc = get_feature_descriptions()
    desc_df = pd.DataFrame(list(desc.items()), columns=["Feature", "Description"])
    st.dataframe(desc_df, use_container_width=True, hide_index=True)
