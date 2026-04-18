"""Check direction encoder classes and probability mapping"""
import joblib
from pathlib import Path

print("=" * 70)
print("NSE DIRECTION ENCODER & PROBABILITY MAPPING CHECK")
print("=" * 70)

# Load direction encoder
encoder_path = Path('data/nse_models/nse_direction_encoder.joblib')
if encoder_path.exists():
    encoder = joblib.load(encoder_path)
    classes = encoder.classes_
    
    print("\n[1] Direction Encoder Classes:")
    print(f"    {classes}")
    print(f"    Order: {list(classes)}")
    
    for i, cls in enumerate(classes):
        print(f"    Index {i}: '{cls}'")
    
    # Check what the indices should be
    if 'Up' in classes:
        up_idx = list(classes).index('Up')
        print(f"\n[2] 'Up' is at index: {up_idx}")
    if 'Down' in classes:
        down_idx = list(classes).index('Down')
        print(f"    'Down' is at index: {down_idx}")
    
    print("\n[3] Expected probability mapping:")
    print(f"    probabilities[:, {up_idx}] = probability of 'Up' → Buy")
    print(f"    probabilities[:, {down_idx}] = probability of 'Down' → Sell")
    
    print("\n[4] Current code in predict_nse_signals.py:")
    print("    up_idx = list(class_names).index('Up') if 'Up' in class_names else 0")
    print("    down_idx = list(class_names).index('Down') if 'Down' in class_names else 1")
    print("    latest_data['buy_probability'] = probabilities[:, up_idx]")
    print("    latest_data['sell_probability'] = probabilities[:, down_idx]")
    
    if up_idx == 0 and down_idx == 1:
        print("\n✅ CORRECT: Up=0, Down=1")
        print("   buy_probability = probabilities[:, 0] = P(Up)")
        print("   sell_probability = probabilities[:, 1] = P(Down)")
    elif up_idx == 1 and down_idx == 0:
        print("\n❌ POTENTIAL ISSUE: Down=0, Up=1")
        print("   buy_probability = probabilities[:, 1] = P(Up) ✅")
        print("   sell_probability = probabilities[:, 0] = P(Down) ✅")
        print("   This is actually CORRECT - classes are alphabetically sorted")
    else:
        print("\n⚠️  UNEXPECTED CLASS ORDER!")
    
else:
    print("\nEncoder file not found!")

# Also check a sample model to verify class order
print("\n" + "=" * 70)
print("VERIFY WITH ACTUAL MODEL")
print("=" * 70)

model_path = Path('data/nse_models/nse_clf_random_forest.joblib')
if model_path.exists():
    model = joblib.load(model_path)
    print(f"\n[5] Random Forest classes_: {model.classes_}")
    print(f"    This should match encoder classes: {classes}")
    if list(model.classes_) == list(classes):
        print("    ✅ Model classes match encoder")
    else:
        print("    ❌ MODEL CLASSES DON'T MATCH ENCODER!")
        print("       This is a CRITICAL BUG!")

print("\n" + "=" * 70)
