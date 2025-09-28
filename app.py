import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import warnings

# Suppress ALL TF/Keras warnings (oneDNN + deprecations)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Quiet TF logs

# Page config
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ Credit Card Fraud Detection App")
st.markdown("Enter transaction features (V1-V28 from PCA, Amount) to predict fraud using trained ML models.")

# Paths
artifacts_dir = "./fraud_detection_artifacts/"
preprocessor_path = os.path.join(artifacts_dir, 'preprocessor.pkl')
lr_path = os.path.join(artifacts_dir, 'lr_model.pkl')
rf_path = os.path.join(artifacts_dir, 'rf_model.pkl')
xgb_path = os.path.join(artifacts_dir, 'xgb_model.pkl')
iso_path = os.path.join(artifacts_dir, 'iso_model.pkl')
nn_path = os.path.join(artifacts_dir, 'nn_model.h5')
smote_path = os.path.join(artifacts_dir, 'smote_sampler.pkl')  # Optional

# Load artifacts
@st.cache_resource(ttl=300)
def load_models():
    loaded = {
        'preprocessor': None,
        'lr_model': None,
        'rf_model': None,
        'xgb_model': None,
        'iso_model': None,
        'nn_model': None,
        'smote': None
    }
    errors = []
    xgb_skipped = False
    
    with st.spinner("Loading models..."):
        # Preprocessor (required)
        try:
            loaded['preprocessor'] = joblib.load(preprocessor_path)
            st.success("✓ Preprocessor loaded")
        except Exception as e:
            errors.append(f"Preprocessor: {e}")
        
        # LR (required)
        try:
            loaded['lr_model'] = joblib.load(lr_path)
            st.success("✓ LR loaded")
        except Exception as e:
            errors.append(f"LR: {e}")
        
        # RF (required)
        try:
            loaded['rf_model'] = joblib.load(rf_path)
            st.success("✓ RF loaded")
        except Exception as e:
            errors.append(f"RF: {e}")
        
        # XGB (optional – skip on sklearn compat error)
        try:
            loaded['xgb_model'] = joblib.load(xgb_path)
            st.success("✓ XGB loaded")
        except ImportError as e:
            if 'parse_version' in str(e):
                xgb_skipped = True
                st.warning("⚠ XGB load skipped (sklearn parse_version compat issue; predictions use 4 models).")
            else:
                errors.append(f"XGB: {e}")
        except Exception as e:
            xgb_skipped = True
            st.warning("⚠ XGB load skipped (pickle issue); predictions use 4 models.")
        
        # ISO (required)
        try:
            loaded['iso_model'] = joblib.load(iso_path)
            st.success("✓ ISO loaded")
        except Exception as e:
            errors.append(f"ISO: {e}")
        
        # NN (required)
        try:
            loaded['nn_model'] = tf.keras.models.load_model(nn_path, compile=False)
            st.success("✓ Neural Network loaded")
        except Exception as e:
            errors.append(f"NN: {e}")
        
        # SMOTE (optional – no import attempt; skip if file missing)
        if os.path.exists(smote_path):
            st.info("ℹ SMOTE file present but skipped (optional for inference).")
        else:
            st.info("ℹ SMOTE file missing (optional for inference; all predictions work).")
        loaded['smote'] = None
    
    if errors:
        st.warning("⚠ Some core models skipped:")
        for err in errors:
            st.error(err)
    
    # Core success: 4 models (exclude XGB; it's optional)
    core_keys = ['preprocessor', 'lr_model', 'rf_model', 'iso_model', 'nn_model']
    core_loaded = all(loaded[k] for k in core_keys)
    if core_loaded:
        st.success("🎉 All core models loaded successfully! (XGB optional)")
    else:
        st.error("❌ Core models missing – check artifacts and sklearn version.")
        st.stop()
    
    return loaded, xgb_skipped

# Load models
models, xgb_load_skipped = load_models()
preprocessor, lr_model, rf_model, xgb_model, iso_model, nn_model, smote = [
    models.get(k) for k in ['preprocessor', 'lr_model', 'rf_model', 'xgb_model', 'iso_model', 'nn_model', 'smote']
]

# Features
feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount']

# Sidebar
st.sidebar.header("Input Transaction Features")
input_data = {}
for name in feature_names:
    if 'V' in name:
        default, min_v, max_v = 0.0, -5.0, 5.0
    else:
        default, min_v, max_v = 100.0, 0.0, 500.0
    input_data[name] = st.sidebar.slider(f"{name}:", min_v, max_v, default, 0.1)

threshold = st.sidebar.slider("Prediction Threshold", 0.3, 0.7, 0.5, 0.05)

# Input prep
X_input = np.array(list(input_data.values())).reshape(1, -1)
X_input_df = pd.DataFrame(X_input, columns=feature_names)

# Predict
if st.sidebar.button("Predict Fraud", type="primary"):
    if not all([preprocessor, lr_model, rf_model, iso_model, nn_model]):
        st.error("❌ Core models not loaded – restart app.")
    else:
        with st.spinner("Preprocessing and predicting..."):
            try:
                X_scaled = preprocessor.transform(X_input_df).astype(np.float32)
                predictions = {}
                xgb_failed = False
                
                # LR
                lr_proba = lr_model.predict_proba(X_scaled)[:, 1][0]
                lr_pred = 1 if lr_proba > threshold else 0
                predictions['Logistic Regression'] = {'pred': lr_pred, 'proba': lr_proba}
                
                # RF
                rf_proba = rf_model.predict_proba(X_scaled)[:, 1][0]
                rf_pred = 1 if rf_proba > threshold else 0
                predictions['Random Forest'] = {'pred': rf_pred, 'proba': rf_proba}
                
                # XGB (skip if not loaded or predict error)
                if xgb_model:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            xgb_proba = xgb_model.predict_proba(X_scaled)[:, 1][0]
                        xgb_pred = 1 if xgb_proba > threshold else 0
                        predictions['XGBoost'] = {'pred': xgb_pred, 'proba': xgb_proba}
                    except Exception as e:
                        xgb_failed = True
                        st.warning(f"⚠ XGBoost predict skipped: {str(e)[:50]}...")
                else:
                    xgb_failed = xgb_load_skipped  # Skip note if load failed
                
                # NN
                nn_proba = nn_model.predict(X_scaled, verbose=0).flatten()[0]
                nn_pred = 1 if nn_proba > threshold else 0
                predictions['Neural Network'] = {'pred': nn_pred, 'proba': nn_proba}
                
                # ISO
                iso_pred_raw = iso_model.predict(X_scaled)[0]
                iso_pred = 1 if iso_pred_raw == -1 else 0
                iso_score = -iso_model.decision_function(X_scaled)[0]
                iso_proba = 1 / (1 + np.exp(-iso_score * 5))
                predictions['Isolation Forest'] = {'pred': iso_pred, 'proba': iso_proba, 'score': iso_score}
                
                # Consensus (4 models)
                probs = [v['proba'] for v in predictions.values() if 'score' not in v and v['proba'] <= 1.0]
                preds = [v['pred'] for v in predictions.values()]
                consensus_pred = 1 if sum(preds) >= len(preds) // 2 + 1 else 0
                consensus_proba = np.mean(probs) if probs else 0
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.stop()

        # Results
        st.header("Predictions")
        col1, col2, col3 = st.columns(3)
        risk_text = "Fraud" if consensus_pred == 1 else "Legitimate"
        risk_level = "🔴 High Risk" if consensus_proba > 0.5 else "🟢 Low Risk"
        with col1: st.metric("Consensus Prediction", risk_text)
        with col2: st.metric("Avg Fraud Probability", f"{consensus_proba:.2%}")
        with col3: st.metric("Risk Level", risk_level)

        # Table (4 models)
        st.subheader("Model Details")
        table_data = []
        for name, data in predictions.items():
            pred_text = 'Fraud' if data['pred'] == 1 else 'Legitimate'
            emoji = '🔴' if data['pred'] == 1 else '🟢'
            if name == 'Isolation Forest':
                row = {
                    'Model': f"{emoji} {name}",
                    'Prediction': pred_text,
                    'Anomaly Score': f"{data['score']:.3f}",
                    'Normalized Proba': f"{data['proba']:.2%}"
                }
            else:
                row = {
                    'Model': f"{emoji} {name}",
                    'Prediction': pred_text,
                    'Fraud Probability': f"{data['proba']:.2%}",
                    'Score': data['proba']
                }
            table_data.append(row)
        results_df = pd.DataFrame(table_data)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Insights
        st.subheader("Insights")
        if consensus_proba < 0.3:
            st.success("🟢 Low risk – Transaction appears legitimate.")
        elif consensus_proba < 0.7:
            st.warning("🟡 Medium risk – Review for elevated fraud indicators.")
        else:
            st.error("🔴 High risk – Potential fraud; block or investigate.")

        xgb_note = " (XGB skipped due to compat – re-save fixes)" if xgb_failed else ""
        st.markdown(f"""
        - **Prediction**: Based on threshold ({threshold:.1f}); 1 = Fraud, 0 = Legitimate{xgb_note}.
        - **Probabilities**: Model confidence (0-1; avg <0.3 = low risk).
        - **Isolation Forest**: Unsupervised anomaly (score >0 = suspicious).
        - **Threshold**: Adjustable in sidebar for sensitivity.
        """)

        # Gauge
        import plotly.graph_objects as go
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=consensus_proba * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fraud Risk (%)", 'font': {'size': 24}},
            delta={'reference': 50, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "black"},
                'bar': {'color': "darkblue", 'thickness': 0.15},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': consensus_proba * 100
                }
            }
        ))
        fig.update_layout(width=500, height=300)
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Models on Kaggle Fraud Dataset | Demo only – not for production.")