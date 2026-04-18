"""Test if VotingClassifier supports sample_weight in sklearn 1.6.1"""
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np

# Create imbalanced dataset
np.random.seed(42)
X = np.random.randn(1000, 10)
y = np.array([0] * 700 + [1] * 300)  # 70% class 0, 30% class 1

print(f"Dataset: {(y==0).sum()} class 0, {(y==1).sum()} class 1")
print(f"Imbalance: {(y==0).sum()/(y==1).sum():.2f}:1")

# Create classifiers
rf = RandomForestClassifier(n_estimators=10, random_state=42)
gb = GradientBoostingClassifier(n_estimators=10, random_state=42)

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb)],
    voting='soft'
)

# Compute sample weights
weights = compute_sample_weight('balanced', y)
print(f"\nSample weights: class 0={weights[y==0][0]:.3f}, class 1={weights[y==1][0]:.3f}")

# Test if fit() accepts sample_weight
try:
    print("\nTrying ensemble.fit(X, y, sample_weight=weights)...")
    ensemble.fit(X, y, sample_weight=weights)
    print("✅ SUCCESS! VotingClassifier supports sample_weight")
    
    # Check if it actually uses the weights
    probs = ensemble.predict_proba(X)
    pred = ensemble.predict(X)
    print(f"\nPredictions: {(pred==0).sum()} class 0, {(pred==1).sum()} class 1")
    print(f"Without balancing, we'd expect ~70% class 0 predictions")
    print(f"With balancing, should be closer to 50/50")
    print(f"Actual: {(pred==0).sum()/(pred==0).sum()+(pred==1).sum()*100:.1f}% class 0")
    
except TypeError as e:
    print(f"❌ ERROR: {e}")
    print("VotingClassifier does NOT support sample_weight in this sklearn version")
