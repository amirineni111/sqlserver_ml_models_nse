# Market-Neutral Features Fix - April 21, 2026

## Problem Summary

**Date**: April 21, 2026
**Symptom**: 98% Sell predictions (42 Buy / 2022 Sell) after retraining with stratified calibration fix
**Root Cause**: Model over-reliance on market context features

## Timeline of Investigation

### Initial Issue (Part 1)
- **9 Buy signals** (99.6% Sell) from biased calibration period
- **Fix Applied**: Stratified calibration split (instead of time-series)
- **Result**: Improved to 42 Buy signals (2% Buy) - still severely biased

### Root Cause Analysis (Part 2)

#### The Discovery
After retraining with stratified calibration, predictions improved from 9 to 42 Buy signals, but this was still only 2% (should be 40-50%).

**Key Finding**: Model's top features were dominated by market context:
1. `dxy_close` (US Dollar strength)
2. `us_10y_yield_close` (Interest rates)
3. `nifty50_return_1d` (Market return)
4. `india_vix_change_pct` (Fear index)
5. `sp500_return_1d` (US Market return)

#### April 21, 2026 Market Conditions
- VIX: 19.5 (rising +3.34%)
- Average VIX last 20 days: **22.7** (elevated - normal < 20)
- S&P 500: down -0.63%
- Dollar: strengthening +0.28%

**The Pattern**: When market-wide indicators are bearish (elevated VIX, negative returns), the model predicts "Sell" for **all stocks**, regardless of individual stock strength.

#### Why This Happened
- Model learned: *"Bearish market conditions → Sell everything"*
- Stock-specific features (RSI, MACD, individual momentum) were underweighted
- Market context features dominated predictions
- No way to identify strong stocks in weak markets

## The Solution: Market-Neutral Features

### Concept
Add features that measure **stock-specific strength RELATIVE to the market**, allowing the model to distinguish:
- Strong stock in weak market → Buy
- Weak stock in strong market → Sell
- Strong stock in strong market → Buy
- Weak stock in weak market → Sell

### 10 New Features Implemented

| Feature | Description | Purpose |
|---------|-------------|---------|
| `stock_return_vs_nifty` | Stock return minus NIFTY return (1-day) | Outperformance metric |
| `stock_return_vs_nifty_5d` | Stock return minus NIFTY return (5-day) | Sustained outperformance |
| `stock_return_vs_sector` | Stock return minus sector average | Peer comparison |
| `rsi_vs_sector_avg` | RSI minus sector average RSI | Relative strength vs peers |
| `volume_vs_sector` | Volume ratio minus sector average | Relative trading activity |
| `volume_anomaly` | Z-score of volume vs 50-day history | Unusual volume detection |
| `beta` | Rolling 60-day beta vs NIFTY | Market sensitivity |
| `beta_adjusted_return` | Return minus expected (beta × market) | Alpha generation |
| `relative_strength_20d` | 20-day cumulative outperformance | Sustained trend |
| `momentum_vs_sector` | 10-day return vs sector momentum | Relative momentum |

### Implementation Details

**Modified Files**:
1. `retrain_nse_model_v2.py` - Added `add_market_neutral_features()` function
2. `predict_nse_signals_v2.py` - Added same function for prediction time

**Calculation Flow**:
```
Load Data → Technical Indicators → Market Context → Market-Neutral Features → Target
```

**Example Calculation** (stock_return_vs_nifty):
```python
# If stock returns +2% and NIFTY returns -1%
stock_return_vs_nifty = 2.0% - (-1.0%) = +3.0%  # Strong outperformance
→ Positive signal for Buy, even though market is down
```

## Expected Results

### Before Fix (With Stratified Calibration Only)
- **42 Buy signals** (2.0%)
- 2022 Sell signals (98.0%)
- Mean P(Buy): 42.7%
- 80.3% of predictions clustered in 0.4-0.5 range

### After Fix (With Market-Neutral Features)
- **800-1000 Buy signals** (40-50%) ← Expected
- Balanced distribution
- Model can identify strong stocks in any market condition
- Predictions less sensitive to market-wide VIX/DXY movements

## Verification Steps

1. **Retrain Model**: `python retrain_nse_model_v2.py`
   - Should pass all 5 validation checks
   - Feature importance should show mix of market-neutral + traditional features

2. **Smoke Test**: `python test_model_artifacts.py`
   - Should show 20-80% Buy range on recent data

3. **Generate Predictions**: `python predict_nse_signals_v2.py`
   - Should produce ~40-50% Buy signals for April 21, 2026
   - Runtime validation should pass (20-80% range)

4. **Analyze Results**: `python diagnose_42_buy_signals.py`
   - Mean P(Buy) should be ~45-55% (not 42.7%)
   - Distribution should be more spread out

## Lessons Learned

### Pattern Recognition
This is the **5th bearish bias bug** in 1 week, each with different root causes:
1. Apr 14: Undefined variables → 0% predictions
2. Apr 16: Missing class balancing → 97.3% Sell
3. Apr 18: VotingClassifier retraining → 97.0% Sell
4. Apr 21 (Part 1): Biased calibration period → 99.6% Sell
5. **Apr 21 (Part 2): Market context over-reliance → 98.0% Sell**

### Key Insight
**Same symptom (bearish bias) ≠ Same root cause**

Each issue required different diagnostic approaches:
- Variable checks (Apr 14)
- Training data analysis (Apr 16, Apr 21 Part 1)
- Model architecture review (Apr 18)
- Feature engineering audit (Apr 21 Part 2)

### Prevention Strategy
1. **Training-time validation** (validate_training_artifacts)
2. **Post-training smoke test** (test_model_artifacts.py)
3. **Runtime validation** (validate_prediction_distribution)
4. **Weekly audit** (audit_model_performance.py - planned)
5. **Feature engineering review** (ensure balanced feature mix)

## Technical Notes

### Why Beta Adjustment Works
```
Expected return = Risk-free rate + Beta × (Market return - Risk-free rate)
Alpha (excess return) = Actual return - Expected return

beta_adjusted_return = stock_return - (beta × market_return)
```
This isolates stock-specific performance from market-wide movements.

### Why Sector Comparisons Matter
Two stocks can both drop 2%, but:
- Stock A dropped 2% while sector dropped 5% → Relative strength (Buy signal)
- Stock B dropped 2% while sector rose 3% → Relative weakness (Sell signal)

### Handling Missing Data
- Sector averages: Calculated only from available stocks on that date
- NIFTY data: Zero-fill returns (neutral), forward-fill levels
- Beta: Default to 1.0 if insufficient history (market-neutral)

## Related Documents
- `CALIBRATION_BIAS_FIX_APRIL_21_2026.md` - Part 1 of the fix (stratified calibration)
- `PREVENTION_STRATEGY_APRIL_21_2026.md` - 4-layer defense system
- `NSE_MODEL_INVESTIGATION_APRIL_2026.md` - Historical bug tracking

## Status
- ✅ Code implemented in retrain_nse_model_v2.py
- ✅ Code implemented in predict_nse_signals_v2.py
- ⏳ Model retraining pending
- ⏳ Validation pending
- ⏳ Production deployment pending

## Next Actions
1. Run full retrain with new features
2. Verify prediction distribution is balanced
3. Monitor first week of predictions for accuracy
4. Document feature importance changes
5. Update NASDAQ pipeline with same fix if needed
