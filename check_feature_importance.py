"""Check model feature importances."""
import sys, json
sys.path.insert(0, '.')
import retrain_nse_model_v2
from retrain_nse_model_v2 import IsotonicCalibratedModel
import __main__
__main__.IsotonicCalibratedModel = IsotonicCalibratedModel
import joblib
import numpy as np

model = joblib.load('data/nse_models/nse_gb_model_v2.joblib')
with open('data/nse_models/selected_features_v2.json') as f:
    feats = json.load(f)

if hasattr(model, 'base_model') and hasattr(model.base_model, 'feature_importances_'):
    fi = model.base_model.feature_importances_
    sorted_fi = sorted(zip(feats, fi), key=lambda x: -x[1])
    print("Feature Importances:")
    for f, imp in sorted_fi:
        print(f"  {f:<35}: {imp:.4f}")
else:
    print("Model structure:")
    print(dir(model))
    if hasattr(model, 'base_model'):
        print("base_model attrs:", dir(model.base_model))

# Also print model metadata
import json
with open('data/nse_models/model_metadata_v2.json') as f:
    meta = json.load(f)
print("\nModel metadata:")
for k, v in meta.items():
    print(f"  {k}: {v}")
