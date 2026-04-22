# NSE Bearish Bias - Recurring Issue Prevention Strategy
**Date:** April 21, 2026  
**Problem:** Same symptom (bearish bias) caused by 4 different bugs in 1 week

---

## THE PATTERN: 4 Different Bugs, Same Symptom

| Date | Root Cause | Symptom | Fix |
|------|-----------|---------|-----|
| **Apr 14** | Undefined variables (`model_names_used`) in prediction code | 0% predictions (silent failure) | Moved variable access inside conditional block |
| **Apr 16** | GradientBoosting trained without class balancing | 97.3% Sell (23 Buy / 2014 Sell) | Added `compute_sample_weight('balanced')` |
| **Apr 18** | VotingClassifier.fit() retrains without sample_weight | 97.0% Sell (21 Buy / 2002 Sell) | Passed `sample_weight=combined_weights` to ensemble |
| **Apr 21** | Calibration set from bearish period (Dec-Feb: 59.5% Down) | 99.6% Sell (9 Buy / 2055 Sell) | Changed to stratified split for calibration |

### Why These Keep Recurring

**Root Problem:** No automated validation pipeline catches these issues before they reach production.

Each fix addresses ONE symptom, but doesn't prevent OTHER ways the same symptom can occur.

---

## PREVENTION STRATEGY: Multi-Layer Defense

### Layer 1: Training-Time Validation (MANDATORY)

Add comprehensive checks in `retrain_nse_model_v2.py` that **ABORT training** if issues detected:

```python
# ============================================================================
# VALIDATION MODULE - Add at end of retrain_nse_model_v2.py
# ============================================================================

def validate_training_artifacts(model, scaler, encoder, X_train, y_train, X_cal, y_cal, X_test, y_test):
    """
    CRITICAL: Validate model artifacts before saving
    If ANY check fails, abort training and alert
    """
    print("\n" + "="*80)
    print("VALIDATION CHECKS (MANDATORY)")
    print("="*80)
    
    issues = []
    
    # CHECK 1: Calibration set balance
    unique, counts = np.unique(y_cal, return_counts=True)
    cal_imbalance = abs(counts[0] - counts[1]) / len(y_cal)
    if cal_imbalance > 0.10:
        issues.append(f"FAIL: Calibration imbalance {cal_imbalance:.1%} > 10%")
    else:
        print(f"✅ CHECK 1: Calibration balance OK ({cal_imbalance:.1%})")
    
    # CHECK 2: Training set balance (should be within 20%)
    unique, counts = np.unique(y_train, return_counts=True)
    train_imbalance = abs(counts[0] - counts[1]) / len(y_train)
    if train_imbalance > 0.20:
        issues.append(f"FAIL: Training imbalance {train_imbalance:.1%} > 20%")
    else:
        print(f"✅ CHECK 2: Training balance OK ({train_imbalance:.1%})")
    
    # CHECK 3: Prediction distribution on test set (should be 20-80% for either class)
    y_test_pred = model.predict(scaler.transform(X_test))
    unique, counts = np.unique(y_test_pred, return_counts=True)
    pred_dist = {encoder.classes_[cls]: count/len(y_test_pred)*100 for cls, count in zip(unique, counts)}
    
    for cls_name, pct in pred_dist.items():
        if pct < 20 or pct > 80:
            issues.append(f"FAIL: Test predictions for '{cls_name}' = {pct:.1f}% (outside 20-80%)")
    
    if not issues:
        print(f"✅ CHECK 3: Test prediction distribution OK")
        for cls_name, pct in pred_dist.items():
            print(f"   {cls_name}: {pct:.1f}%")
    
    # CHECK 4: Probability calibration sanity
    y_test_proba = model.predict_proba(scaler.transform(X_test))
    avg_proba = y_test_proba.mean(axis=0)
    
    for i, cls_name in enumerate(encoder.classes_):
        if avg_proba[i] < 0.25 or avg_proba[i] > 0.75:
            issues.append(f"FAIL: Avg probability for '{cls_name}' = {avg_proba[i]:.2%} (outside 25-75%)")
    
    if not issues:
        print(f"✅ CHECK 4: Probability calibration OK")
        for i, cls_name in enumerate(encoder.classes_):
            print(f"   Avg P({cls_name}): {avg_proba[i]:.2%}")
    
    # CHECK 5: Model artifact integrity
    try:
        # Test serialization
        import io
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        test_model = joblib.load(buffer)
        
        # Test prediction
        test_pred = test_model.predict(scaler.transform(X_test[:10]))
        print(f"✅ CHECK 5: Model serialization OK")
    except Exception as e:
        issues.append(f"FAIL: Model serialization error: {e}")
    
    # FINAL VERDICT
    print("\n" + "="*80)
    if issues:
        print("❌ VALIDATION FAILED - ABORTING TRAINING")
        print("="*80)
        for issue in issues:
            print(f"  • {issue}")
        print("\nModel NOT saved. Fix issues and retrain.")
        sys.exit(1)
    else:
        print("✅ ALL VALIDATION CHECKS PASSED")
        print("="*80)
        return True


# Modify main() to call validation BEFORE saving:
def main():
    # ... existing training code ...
    
    # Train model
    model, base_model, results = train_model(
        X_train_scaled, y_train,
        X_cal_scaled, y_cal,
        X_test_scaled, y_test
    )
    
    # ⭐ CRITICAL: VALIDATE BEFORE SAVING
    validate_training_artifacts(
        model, scaler, encoder,
        X_train_scaled, y_train,
        X_cal_scaled, y_cal,
        X_test_scaled, y_test
    )
    
    # Only save if validation passed
    save_model_artifacts(model, base_model, scaler, encoder, selected_features, importances)
```

---

### Layer 2: Post-Training Smoke Test (AUTOMATED)

Create `test_model_artifacts.py` that runs automatically after training:

```python
"""
Post-training smoke test
Run automatically after retrain to catch issues before production
"""

def test_model_predictions():
    """Test model on recent data to verify prediction distribution"""
    
    # Load latest model
    model = joblib.load('data/nse_models/nse_gb_model_v2.joblib')
    scaler = joblib.load('data/nse_models/nse_scaler_v2.joblib')
    encoder = joblib.load('data/nse_models/nse_direction_encoder_v2.joblib')
    
    # Get last 5 days of data
    conn = get_db_connection()
    df = load_recent_data(conn, days=5)
    
    # Predict
    predictions = predict_for_tickers(df, model, scaler, encoder)
    
    # Check distribution
    buy_pct = len(predictions[predictions['predicted_signal'] == 'Buy']) / len(predictions) * 100
    sell_pct = 100 - buy_pct
    
    print(f"\n{'='*80}")
    print(f"SMOKE TEST: Prediction Distribution (Last 5 Days)")
    print(f"{'='*80}")
    print(f"Buy:  {buy_pct:.1f}%")
    print(f"Sell: {sell_pct:.1f}%")
    
    # FAIL if distribution is extreme
    if buy_pct < 20 or buy_pct > 80:
        print(f"\n❌ SMOKE TEST FAILED: {buy_pct:.1f}% Buy is outside acceptable range (20-80%)")
        print(f"Model may be biased. Review training logs.")
        sys.exit(1)
    else:
        print(f"\n✅ SMOKE TEST PASSED")
        return True

if __name__ == '__main__':
    test_model_predictions()
```

---

### Layer 3: Production Monitoring (RUNTIME)

Add to `predict_nse_signals_v2.py`:

```python
def validate_prediction_distribution(predictions_df):
    """
    Validate prediction distribution before writing to database
    Alert if distribution is abnormal
    """
    total = len(predictions_df)
    buy_count = len(predictions_df[predictions_df['predicted_signal'] == 'Buy'])
    sell_count = total - buy_count
    
    buy_pct = buy_count / total * 100
    sell_pct = sell_count / total * 100
    
    print(f"\n{'='*80}")
    print(f"PREDICTION DISTRIBUTION VALIDATION")
    print(f"{'='*80}")
    print(f"Total: {total:,}")
    print(f"Buy:   {buy_count:,} ({buy_pct:.1f}%)")
    print(f"Sell:  {sell_count:,} ({sell_pct:.1f}%)")
    
    # CRITICAL: Alert if extreme skew
    if buy_pct < 20 or buy_pct > 80:
        print(f"\n⚠️  WARNING: EXTREME PREDICTION SKEW DETECTED")
        print(f"   Buy: {buy_pct:.1f}% is outside normal range (20-80%)")
        print(f"   This may indicate model bias or calibration issue")
        print(f"   Review model training and consider retraining with stratified calibration")
        
        # Write alert to database
        alert_msg = f"NSE prediction skew: {buy_pct:.1f}% Buy on {predictions_df['trading_date'].iloc[0]}"
        # conn.execute(f"INSERT INTO ml_alerts (date, message, severity) VALUES (GETDATE(), '{alert_msg}', 'HIGH')")
        
        # OPTION: Abort prediction write and force manual review
        # return False
    
    return True

# Modify main() to call validation:
def main():
    # ... generate predictions ...
    
    # Validate before writing
    if not validate_prediction_distribution(predictions):
        print("[ERROR] Prediction validation failed. Aborting database write.")
        sys.exit(1)
    
    # Write to database
    write_predictions_to_database(predictions)
```

---

### Layer 4: Weekly Model Audit (MANUAL)

Create `audit_model_performance.py` - run weekly:

```python
"""
Weekly model audit - checks historical prediction accuracy and drift
"""

def audit_model():
    """Audit model performance over last 30 days"""
    
    conn = get_db_connection()
    
    # Check prediction distribution over time
    query = """
    SELECT 
        trading_date,
        COUNT(*) as total,
        SUM(CASE WHEN predicted_signal = 'Buy' THEN 1 ELSE 0 END) as buy_count,
        CAST(SUM(CASE WHEN predicted_signal = 'Buy' THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100 AS DECIMAL(5,2)) as buy_pct
    FROM ml_nse_trading_predictions
    WHERE trading_date >= DATEADD(day, -30, CAST(GETDATE() AS DATE))
    GROUP BY trading_date
    ORDER BY trading_date DESC
    """
    
    results = conn.execute_query(query)
    
    # Flag days with extreme skew
    extreme_days = results[(results['buy_pct'] < 20) | (results['buy_pct'] > 80)]
    
    if len(extreme_days) > 0:
        print(f"\n⚠️  AUDIT ALERT: {len(extreme_days)} days with extreme prediction skew:")
        print(extreme_days[['trading_date', 'buy_pct']].to_string(index=False))
        print(f"\nRecommendation: Review model calibration and consider retraining")
    else:
        print(f"\n✅ AUDIT PASSED: All days within acceptable range")
    
    return extreme_days
```

---

## IMPLEMENTATION PLAN

### Immediate (Today)
1. ✅ Fix calibration split (DONE)
2. Add validation checks to `retrain_nse_model_v2.py`
3. Create `test_model_artifacts.py`
4. **RETRAIN MODEL** with validation enabled

### This Week
1. Add runtime distribution checks to `predict_nse_signals_v2.py`
2. Create `audit_model_performance.py`
3. Set up weekly audit email alerts

### Ongoing
1. Run smoke test after every retrain
2. Monitor prediction distribution daily
3. Weekly audit review

---

## CRITICAL: Update Retraining Workflow

### OLD (Fragile):
```bash
python retrain_nse_model_v2.py  # Pray it works
python predict_nse_signals_v2.py  # Hope for the best
```

### NEW (Robust):
```bash
# 1. Train with validation
python retrain_nse_model_v2.py
# Script exits with code 1 if validation fails

# 2. Smoke test (automatic)
python test_model_artifacts.py
# Script exits with code 1 if predictions look wrong

# 3. Only if both pass, deploy
python predict_nse_signals_v2.py
# Runtime validation alerts if skew detected

# 4. Weekly audit
python audit_model_performance.py  # Every Sunday after retrain
```

---

## EXPECTED OUTCOME

With these 4 layers of defense:

1. **Layer 1** catches calibration issues DURING training
2. **Layer 2** catches prediction issues BEFORE production
3. **Layer 3** catches runtime issues and alerts immediately
4. **Layer 4** catches drift over time

**No single bug can cause 99% Sell predictions to reach production.**

---

## FILES TO CREATE/MODIFY

### Create New:
1. `test_model_artifacts.py` - Post-training smoke test
2. `audit_model_performance.py` - Weekly audit script

### Modify Existing:
1. `retrain_nse_model_v2.py` - Add `validate_training_artifacts()` function
2. `predict_nse_signals_v2.py` - Add `validate_prediction_distribution()` function
3. `run_weekly_retrain.bat` - Add smoke test after retrain

---

**Next Action:** Implement Layer 1 (training validation) NOW, before retraining.
