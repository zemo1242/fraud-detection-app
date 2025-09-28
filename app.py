import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Suppress oneDNN TF warnings (optional; faster on CPU)
warnings.filterwarnings('ignore', category=DeprecationWarning)  # Suppress Keras/TF deprecation noise

# Page config (title, layout)
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ Credit Card Fraud Detection App")
st.markdown("Enter transaction features (V1-V28 from PCA, Amount) to predict fraud using trained ML models.")

# Paths (update to your local artifacts folder)
artifacts_dir = "./fraud_detection_artifacts/"  # Change if different (e.g., '../artifacts/')
preprocessor_path = os.path.join(artifacts_dir, 'preprocessor.pkl')
lr_path = os.path.join(artifacts_dir, 'lr_model.pkl')
rf_path = os.path.join(artifacts_dir, 'rf_model.pkl')
xgb_path = os.path.join(artifacts_dir, 'xgb_model.pkl')
iso_path = os.path.join(artifacts_dir, 'iso_model.pkl')
nn_path = os.path.join(artifacts_dir, 'nn_model.h5')
smote_path = os.path.join(artifacts_dir, 'smote_sampler.pkl')  # Optional

# Load artifacts (with error handling)
@st.cache_resource(ttl=300)  # 5 min cache refresh for dev
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
    
    with st.spinner("Loading models..."):
        # Core preprocessor (required for most)
        try:
            loaded['preprocessor'] = joblib.load(preprocessor_path)
            st.success("✓ Preprocessor loaded")
        except Exception as e:
            errors.append(f"Preprocessor: {e}")
        
        # Load each model individually (graceful if one fails)
        for model_name, path in [
            ('lr_model', lr_path),
            ('rf_model', rf_path),
            ('xgb_model', xgb_path),
            ('iso_model', iso_path),
            ('nn_model', nn_path)
        ]:
            if model_name == 'nn_model':
                try:
                    loaded[model_name] = tf.keras.models.load_model(path, compile=False)
                    st.success("✓ Neural Network loaded")
                except Exception as e:
                    errors.append(f"NN: {e}")
            else:
                try:
                    loaded[model_name] = joblib.load(path)
                    st.success(f"✓ {model_name.replace('_model', '').upper()} loaded")
                except Exception as e:
                    errors.append(f"{model_name}: {e}")
        
        # SMOTE optional (for training only; skip for inference)
        if os.path.exists(smote_path):
            try:
                loaded['smote'] = joblib.load(smote_path)
                st.success("✓ SMOTE loaded (optional)")
            except Exception as e:
                st.warning(f"ℹ SMOTE load skipped: {e}")
        else:
            st.info("ℹ SMOTE missing (optional for inference; all predictions work)")
    
    if errors:
        st.warning("⚠ Some models skipped (app continues with available):")
        for err in errors:
            st.error(err)
        st.info("Fix: Re-save models in Colab with matching versions (sklearn 1.6.1).")
    
    core_loaded = all(loaded[k] for k in loaded if k != 'smote')
    if core_loaded:
        st.success("🎉 All core models loaded successfully!")
    else:
        st.error("❌ Core models missing – check artifacts folder.")
        st.stop()
    
    return loaded

models = load_models()
preprocessor, lr_model, rf_model, xgb_model, iso_model, nn_model, smote = [
    models.get(k) for k in ['preprocessor', 'lr_model', 'rf_model', 'xgb_model', 'iso_model', 'nn_model', 'smote']
]

# Feature names (for input form)
feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount']  # V1-V28, Amount
n_features = len(feature_names)

# Sidebar: Input form (sliders for realistic ranges; V1-V28: -5 to 5, Amount: 0-500)
st.sidebar.header("Input Transaction Features")
input_data = {}
for name in feature_names:
    if 'V' in name:  # PCA features: ~mean 0, std 1 (input pre-scale)
        default = 0.0
        min_value, max_value = -5.0, 5.0
    else:  # Amount: 0-500 (capped)
        default = 100.0
        min_value, max_value = 0.0, 500.0
    input_data[name] = st.sidebar.slider(f"{name}:", min_value=min_value, max_value=max_value, value=default, step=0.1)

# Adjustable threshold (for pred decisions)
threshold = st.sidebar.slider("Prediction Threshold", 0.3, 0.7, 0.5, 0.05)

# Convert to array
X_input = np.array(list(input_data.values())).reshape(1, -1)  # Shape: (1, 29)
X_input_df = pd.DataFrame(X_input, columns=feature_names)

# Predict button
if st.sidebar.button("Predict Fraud", type="primary"):
    if not all([preprocessor, lr_model, rf_model, iso_model, nn_model]):
        st.error("❌ Core models not loaded – check warnings above.")
    else:
        with st.spinner("Preprocessing and predicting..."):
            try:
                # Step 1: Preprocess (scale for all models)
                if preprocessor:
                    X_scaled = preprocessor.transform(X_input_df).astype(np.float32)  # TF needs float32
                else:
                    X_scaled = X_input  # Fallback
                
                # Step 2: Predictions from all models
                predictions = {}
                xgb_failed = False
                
                # Logistic Regression (scaled)
                if lr_model:
                    lr_proba = lr_model.predict_proba(X_scaled)[:, 1][0]
                    lr_pred = 1 if lr_proba > threshold else 0
                    predictions['Logistic Regression'] = {'pred': lr_pred, 'proba': lr_proba}
                
                # Random Forest (scaled)
                if rf_model:
                    rf_proba = rf_model.predict_proba(X_scaled)[:, 1][0]
                    rf_pred = 1 if rf_proba > threshold else 0
                    predictions['Random Forest'] = {'pred': rf_pred, 'proba': rf_proba}
                
                # XGBoost (scaled – assuming fit on scaled data; wrap to handle deprecation)
                if xgb_model:
                    try:
                        # Temp suppress warnings for XGBoost
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            xgb_proba = xgb_model.predict_proba(X_scaled)[:, 1][0]  # Use scaled (consistent)
                        xgb_pred = 1 if xgb_proba > threshold else 0
                        predictions['XGBoost'] = {'pred': xgb_pred, 'proba': xgb_proba}
                    except Exception as xgb_e:
                        xgb_failed = True
                        st.warning(f"⚠ XGBoost predict skipped (old pickle deprecation: {xgb_e}). Using 4 models.")
                
                # Neural Network (scaled)
                if nn_model:
                    nn_proba = nn_model.predict(X_scaled, verbose=0).flatten()[0]
                    nn_pred = 1 if nn_proba > threshold else 0
                    predictions['Neural Network'] = {'pred': nn_pred, 'proba': nn_proba}
                
                # Isolation Forest (scaled; anomaly detection)
                if iso_model:
                    iso_pred_raw = iso_model.predict(X_scaled)[0]
                    iso_pred = 1 if iso_pred_raw == -1 else 0
                    iso_score = -iso_model.decision_function(X_scaled)[0]  # Higher = more anomalous
                    # Normalize to 0-1 (sigmoid approx; scale factor based on typical ISO ranges)
                    iso_proba = 1 / (1 + np.exp(-iso_score * 5))  # Sigmoid: 0 legit, 1 fraud
                    predictions['Isolation Forest'] = {'pred': iso_pred, 'proba': iso_proba, 'score': iso_score}
                
                # Consensus (majority vote on preds; avg proba from probabilistic models only)
                if predictions:
                    probs = [v['proba'] for v in predictions.values() if 'score' not in v]  # Exclude ISO
                    preds = [v['pred'] for v in predictions.values()]
                    consensus_pred = 1 if sum(preds) >= len(preds) // 2 + 1 else 0  # Majority
                    consensus_proba = np.mean(probs) if probs else 0
                else:
                    consensus_pred, consensus_proba = 0, 0
                
            except Exception as e:
                st.error(f"Prediction failed (non-XGBoost): {e}")
                st.info("Check input shapes or model compatibility.")
                st.stop()
    
        # Display Results
        st.header("Predictions")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        risk_text = "Fraud" if consensus_pred == 1 else "Legitimate"
        color = "🔴 High Risk" if consensus_proba > 0.5 else "🟢 Low Risk"
        with col1:
            st.metric("Consensus Prediction", risk_text)
        with col2:
            st.metric("Avg Fraud Probability", f"{consensus_proba:.2%}")
        with col3:
            st.metric("Risk Level", color)
        
        # Detailed table
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
            st.warning("🟡 Medium risk – Review for elevated fraud indicators (e.g., high Amount, outlier V features).")
        else:
            st.error("🔴 High risk – Potential fraud; block or investigate.")
        
        xgb_note = " (XGBoost skipped due to old pickle – re-save fixes)" if xgb_failed else ""
        st.write("""
        - **Prediction**: Based on threshold ({:.1f}); 1 = Fraud, 0 = Legitimate{}.
        - **Probabilities**: Model confidence (0-1; avg <0.3 = low risk).
        - **Isolation Forest**: Unsupervised anomaly detection (score >0 = suspicious; normalized for table).
        - **Best Model**: XGBoost/RF (handle imbalance well){}.
        - **Threshold**: Adjustable in sidebar – lower for more sensitive detection.
        """.format(threshold, xgb_note, xgb_note))
        
        # Plot: Fraud Risk Gauge
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
st.markdown("Built with Streamlit | Models trained on Kaggle Credit Card Fraud Dataset | Threshold adjustable for precision/recall. For demo only – not for production fraud detection.")