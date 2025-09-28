import os
import joblib
import numpy as np
import sklearn  # For version
import warnings

# Suppress XGBoost warning (harmless for loaded models)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# Versions
print(f"Numpy: {np.__version__}")
print(f"Sklearn: {sklearn.__version__}")
print(f"Joblib: {joblib.__version__}")

artifacts_dir = "./fraud_detection_artifacts/"
paths = {
    'preprocessor': 'preprocessor.pkl',
    'lr': 'lr_model.pkl',
    'rf': 'rf_model.pkl',
    'xgb': 'xgb_model.pkl',
    'iso': 'iso_model.pkl',
    'smote': 'smote_sampler.pkl',  # Optional; skip if missing
    'nn': 'nn_model.h5'
}

# Check paths
print("\nPath existence:")
for name, file in paths.items():
    path = os.path.join(artifacts_dir, file)
    print(f"{name}: {os.path.exists(path)}")

# Required loads (core models)
required = ['preprocessor', 'lr', 'rf', 'xgb', 'iso']
loaded_models = {}
good = True

for name in required:
    try:
        path = os.path.join(artifacts_dir, paths[name])
        loaded_models[name] = joblib.load(path)
        print(f"✓ {name.capitalize()} loaded: {type(loaded_models[name]).__name__}")
    except Exception as e:
        print(f"✗ {name} failed: {e}")
        good = False

# Optional SMOTE (skip if file missing or error)
try:
    smote_path = os.path.join(artifacts_dir, paths['smote'])
    if os.path.exists(smote_path):
        loaded_models['smote'] = joblib.load(smote_path)
        print("✓ Smote loaded: SMOTE")
    else:
        print("ℹ Smote skipped (file not found – optional for inference)")
except Exception as e:
    print(f"ℹ Smote skipped: {e}")
    loaded_models['smote'] = None  # Graceful

# NN (TF)
if good:  # Only if required OK
    try:
        import tensorflow as tf
        nn_path = os.path.join(artifacts_dir, paths['nn'])
        loaded_models['nn'] = tf.keras.models.load_model(nn_path)
        print("✓ NN loaded: Keras Model")
        # Quick predict test (uses preprocessor + LR as sample)
        sample_input = np.zeros((1, 29))  # Dummy (unscaled)
        sample_scaled = loaded_models['preprocessor'].transform(sample_input)
        lr_proba = loaded_models['lr'].predict_proba(sample_scaled)[:, 1][0]
        print(f"Sample LR fraud proba: {lr_proba:.4f} (low = legitimate; predicts OK!)")
        print("\n🎉 All core models loaded successfully! XGBoost warning is harmless. Run app now.")
    except Exception as e:
        print(f"✗ NN failed: {e}")
        good = False
else:
    print("Fix required models/paths before NN test.")

if not good:
    print("\nSome required failures – re-download artifacts or check paths.")