# NSE V2 Feature Weighting Analysis
## April 18, 2026

---

## CURRENT FEATURE IMPORTANCE (Unbalanced)

The current V2 model has **severely imbalanced** feature weights:

| Feature | Importance | % of Total | Category |
|---------|-----------|-----------|----------|
| **return_5d** | **0.3486** | **34.9%** | **Price Momentum** |
| bb_position | 0.1185 | 11.9% | Volatility |
| rsi | 0.0806 | 8.1% | Momentum |
| return_10d | 0.0697 | 7.0% | Price Momentum |
| price_to_sma20 | 0.0538 | 5.4% | Moving Average |
| return_20d | 0.0362 | 3.6% | Price Momentum |
| price_to_sma50 | 0.0276 | 2.8% | Moving Average |
| nifty50_return_1d | 0.0147 | 1.5% | Market Context |
| **Top 8 features** | **0.769** | **77%** | **Dominance** |

### Problem Identified

1. **return_5d dominates with 35% weight** - This is the 5-day return, which makes the model heavily dependent on recent price momentum
2. **Top 3 features control 55% of prediction** - Model is not well-balanced
3. **Market context features are underweighted** - nifty50_return_1d only 1.5%, india_vix_close 1.3%

This extreme weighting means the model is essentially asking: **"What did the price do in the last 5-10 days?"** rather than considering broader market conditions and fundamentals.

---

## WHY THIS IS A PROBLEM

### Issue #1: Overfitting to Recent Price Action
- The model learns "stocks that went up keep going up"
- Ignores market regime changes, sentiment shifts, volatility spikes
- Performs poorly when momentum reverses

### Issue #2: Ignoring Macro Context
- India VIX (fear gauge): Only 1.3% weight
- NIFTY 50 market direction: Only 1.5% weight
- These should matter more for predicting future moves!

### Issue #3: Data Leakage Risk
- return_5d is calculated from past 5 days
- But we're predicting direction_5d (next 5 days)
- Strong correlation exists but may not be predictive

---

## SOLUTION: NATURAL FACTOR WEIGHTING

### Approach 1: Feature Engineering (Recommended)

**Remove or reduce dominant features:**
```python
# Current: return_5d = (close - close_5d_ago) / close_5d_ago
# Problem: 35% weight, dominates model

# Option A: Remove return_5d entirely
# Let model learn from RSI, MACD, BB instead

# Option B: Replace with normalized returns
# Divide by volatility to reduce outlier impact
return_5d_normalized = return_5d / atr_pct

# Option C: Cap feature importance
# Use max_features or feature subsampling in GB model
```

**Boost macro feature importance:**
```python
# Create composite macro features
market_stress = india_vix_close / india_vix_close.rolling(20).mean()
market_momentum = nifty50_return_1d * volume_ratio

# These combine multiple signals → higher predictive value
```

### Approach 2: Ensemble Rebalancing

**Use feature subsampling:**
```python
GradientBoostingClassifier(
    max_features=0.5,  # Randomly use 50% of features per tree
    # Forces model to learn from diverse features
    # Reduces dominance of any single feature
)
```

**Train sector-specific models:**
```python
# Banking stocks behave differently than Tech stocks
# Train 11 separate models (one per sector)
# Aggregate predictions with equal weighting
# Each sector model learns its own patterns
```

### Approach 3: Manual Feature Weighting

**Pre-process features with domain knowledge:**
```python
# Define feature groups and desired weights
weights = {
    'price_momentum': 0.25,    # Down from 50%
    'technical': 0.30,         # RSI, MACD, BB
    'volume': 0.15,
    'market_context': 0.20,    # Up from 3%!
    'volatility': 0.10
}

# Scale features to achieve target distribution
# Before training, multiply each feature by group weight
```

---

## RECOMMENDED CHANGES FOR V3

### Phase 1: Remove Dominant Features
```python
# In retrain_nse_model_v2.py, line 430-444 (select_features)

# BEFORE: Select top 20 by importance
top_features = feature_importance.head(20)['feature'].tolist()

# AFTER: Select top 30, then exclude over-dominant ones
top_features = feature_importance.head(30)['feature'].tolist()

# Remove return_5d, return_10d, return_20d (momentum bias)
excluded_features = ['return_5d', 'return_10d', 'return_20d']
balanced_features = [f for f in top_features if f not in excluded_features][:20]
```

### Phase 2: Add Feature Subsampling
```python
# In retrain_nse_model_v2.py, line 446-520 (train_model)

GB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'max_features': 0.6,  # ADD THIS: Use 60% of features per tree
    'random_state': 42
}
```

### Phase 3: Increase Market Context Features
```python
# In retrain_nse_model_v2.py, line 360-400 (merge_market_context)

# ADD these composite features:
df['market_stress'] = df['india_vix_close'] / df['india_vix_close'].rolling(20).mean()
df['market_regime'] = np.where(df['nifty50_return_1d'] > 0, 1, -1)
df['macro_momentum'] = df['nifty50_return_1d'] * df['volume_ratio']
```

---

## EXPECTED IMPROVEMENTS

### With Balanced Features:
- **Accuracy:** May drop 2-3% initially (77% → 74%)
- **Robustness:** Significantly better across different market conditions
- **Signal quality:** More balanced Buy/Sell ratio (not 99% one direction)
- **Generalization:** Better performance on unseen data (post-training period)

### Trade-off Analysis:
```
Current V2:
✅ High accuracy (77.5%) on test set
❌ Extreme predictions (98% Buy or 98% Sell)
❌ Overfits to price momentum
❌ Breaks when fundamentals change

Balanced V3:
✅ Realistic predictions (60-65% Buy in bull market)
✅ Considers macro conditions
✅ More stable across market regimes
⚠️ Slightly lower test accuracy (trade-off for robustness)
```

---

## NEXT STEPS

1. **Let current V2 retrain complete** (with extended data through April 15)
2. **Evaluate V2 results** - Check if predictions are more balanced now
3. **If still extreme (>95% one direction):**
   - Implement Phase 1 (remove dominant features)
   - Implement Phase 2 (feature subsampling)
4. **Track performance** over 2-3 weeks of daily predictions
5. **Consider V3 rebuild** with all three phases if needed

---

## CONCLUSION

**The "error on market_cap_category"** was a red herring - it's not even in the top 20 features!

**The real issue:** return_5d has 35% weight, causing extreme momentum bias.

**The fix:** Retrain with balanced features that naturally weight:
- 25% price patterns (not just momentum)
- 30% technical indicators
- 20% market context (up from 3%!)
- 15% volume signals
- 10% volatility

This creates a **naturally balanced model** that considers the full picture, not just "what happened yesterday."
