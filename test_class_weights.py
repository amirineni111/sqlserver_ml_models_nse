"""Quick test to verify class weight calculation"""
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

# Simulate NSE training data distribution
n_samples = 537111
down_pct = 0.546
up_pct = 0.454

# Create fake y_train
n_down = int(n_samples * down_pct)
n_up = n_samples - n_down

y_train = np.array([0] * n_down + [1] * n_up)
np.random.shuffle(y_train)

print(f"Training data:")
print(f"  Total: {len(y_train)}")
print(f"  Down (0): {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.1f}%)")
print(f"  Up (1): {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")

# Compute balanced weights
class_weights = compute_sample_weight('balanced', y_train)

print(f"\nClass weights:")
print(f"  Down weight: {class_weights[y_train==0][0]:.6f}")
print(f"  Up weight: {class_weights[y_train==1][0]:.6f}")
print(f"  Ratio (Up/Down): {class_weights[y_train==1][0] / class_weights[y_train==0][0]:.3f}")

# Expected formula: n_samples / (n_classes * bincount)
expected_down = n_samples / (2 * n_down)
expected_up = n_samples / (2 * n_up)
print(f"\nExpected weights (manual calculation):")
print(f"  Down weight: {expected_down:.6f}")
print(f"  Up weight: {expected_up:.6f}")
print(f"  Ratio (Up/Down): {expected_up / expected_down:.3f}")
