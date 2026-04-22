# HYBRID APPROACH - April 21, 2026

## Executive Summary

After **5 iterations of bearish bias fixes** (each from a different root cause), we discovered the fundamental issue: **market context features are VALUABLE but were OVER-DOMINANT**.

The model learned: `if VIX > 20: return "Sell" for ALL stocks`

We needed: `if VIX > 20 AND stock_weak: "Sell" | if VIX > 20 AND stock_strong: "Buy"`

---

## The Problem Evolution

| Date | Symptom | Root Cause | Fix |
|------|---------|-----------|-----|
| Apr 14 | 0% predictions | Undefined variables | Added variable definitions |
| Apr 16 | 97.3% Sell | No class balancing | Added `class_weight='balanced'` |
| Apr 18 | 97.0% Sell | VotingClassifier retraining bug | Removed VotingClassifier |
| Apr 21 (Part 1) | 99.6% Sell (9 Buy) | Biased calibration set | Stratified calibration split |
| Apr 21 (Part 2) | 98.0% Sell (42 Buy) | Market feature dominance | **THIS FIX - Hybrid Approach** |

---

## Root Cause Analysis (Part 2)

### Feature Importance Analysis (Bad Model - 42 Buy signals)

| Rank | Feature | Importance | Category |
|------|---------|-----------|----------|
| 1 | nifty50_close | 9.3% | Market Context |
| 2 | vix_close | 8.0% | Market Context |
| 3 | dxy_close | 6.5% | Market Context |
| 4 | relative_strength_20d | 5.5% | Market-Neutral |
| 5 | us_10y_yield_close | 5.4% | Market Context |

**Problem**: 4 of top 5 features are market context → Model learns market regime, not stock selection

### April 21 Market Conditions
- **VIX**: 19.5 (+3.34%), 20-day avg: 22.7 (elevated)
- **S&P 500**: -0.63% (slightly down)
- **DXY**: +0.28% (strengthening dollar)
- **Model Logic**: VIX elevated + S&P down + Dollar up = **"Sell everything"**

### Prediction Distribution
- Mean P(Buy): 42.7% (clustered below 0.5 threshold)
- Only 42 predictions exceeded 0.5 (2% of 2,064 stocks)
- 80.3% of predictions: 0.40 - 0.50 range (just below threshold)

**Insight**: Model wanted to Buy more stocks but market context suppressed probabilities

---

## Solution: Three-Pronged Hybrid Approach

### 1. MARKET-NEUTRAL FEATURES (10 features)

Measure stock-specific strength **relative** to market/sector:

| Feature | Formula | Purpose |
|---------|---------|---------|
| `stock_return_vs_nifty` | Stock return - NIFTY return | Outperformance vs benchmark |
| `stock_return_vs_nifty_5d` | 5-day outperformance | Sustained strength |
| `stock_return_vs_sector` | Stock return - sector avg | Peer comparison |
| `rsi_vs_sector_avg` | Stock RSI - sector avg RSI | Relative momentum |
| `volume_vs_sector` | Stock volume - sector avg | Unusual activity |
| `volume_anomaly` | Z-score vs 50-day history | Stock-specific spikes |
| `beta` | 60-day rolling beta | Market sensitivity |
| `beta_adjusted_return` | Return - (beta × market return) | Risk-adjusted performance |
| `relative_strength_20d` | 20-day cumulative outperformance | Trend strength |
| `momentum_vs_sector` | 10-day return vs sector | Sector leadership |

**Impact**: Allows model to identify Stock A (+3%) > Stock B (+1%) even when market is down -2%

---

### 2. INTERACTION FEATURES (10 features)

Explicitly combine market context WITH stock-specific signals:

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `outperformance_in_fear` | stock_vs_nifty × (VIX / 15) | **High VIX + Outperforming = Strong Buy** |
| `outperformance_in_greed` | stock_vs_nifty × (15 / VIX) | Low VIX + Outperforming = Momentum |
| `sector_leader_conviction` | (RSI_vs_sector / 20) × volume_anomaly | Quality + Volume = High conviction |
| `defensive_risk_score` | beta × (VIX / 15) | High beta in fear = Risky |
| `quality_opportunity` | stock_vs_nifty / beta | Low beta outperformers = Quality |
| `risk_adjusted_momentum` | stock_vs_nifty_5d / (ATR / price) | Return per unit of risk |
| `global_market_sync` | stock_vs_nifty × SP500_return | Coordinated with global markets? |
| `contrarian_strength` | stock_vs_nifty × (-NIFTY_return) | **Up when market down = Winner** |
| `quality_in_volatility` | RSI_vs_sector × (VIX / VIX_20d_avg) | Strong during volatility spikes |
| `sector_momentum_confirmed` | momentum_vs_sector × NIFTY_return | Sector leadership with market confirmation |

**Key Examples**:
- **Contrarian strength**: Stock +2%, Market -3% → Score = +2% × 3% = +0.06 (positive signal)
- **Outperformance in fear**: Stock +1%, VIX 30 → Score = +1% × (30/15) = +2% (amplified Buy)
- **Defensive risk**: Beta 1.5, VIX 30 → Score = 1.5 × 2.0 = 3.0 (high risk - avoid)

---

### 3. WEIGHTED FEATURE SELECTION

Force balanced representation across categories:

| Category | Target % | Target Count (of 20) | Purpose |
|----------|----------|---------------------|---------|
| **Market Context** | 25% | 5 features | Keep VIX/DXY/yields but **limited** |
| **Stock-Specific** | 35% | 7 features | Core technical/fundamental signals |
| **Relative/Neutral** | 25% | 5 features | Comparative metrics |
| **Interactions** | 15% | 3 features | Smart combinations |

**Mechanism**:
1. Calculate feature importance for ALL features
2. Categorize features into 4 groups
3. Select top N from EACH category (not global top N)
4. Ensures no category dominates (max 35% for any category)

**Example Bad Selection (Before)**:
- Market: 60% (12 features) ← **PROBLEM**
- Stock: 20% (4 features)
- Relative: 15% (3 features)
- Interactions: 5% (1 feature)

**Example Good Selection (After)**:
- Market: 25% (5 features) ← **Controlled**
- Stock: 35% (7 features) ← **Prioritized**
- Relative: 25% (5 features)
- Interactions: 15% (3 features)

---

## Implementation Details

### Training Script (`retrain_nse_model_v2.py`)

**Added Functions**:
1. `add_market_neutral_features(df)` - Line 454
2. `add_interaction_features(df)` - Line 587
3. `categorize_features(feature_list)` - Line 621
4. `select_features(X, y)` - Line 668 (modified)

**Feature Engineering Pipeline**:
```python
df = calculate_technical_indicators(df)  # 45 features
df = merge_market_context(df)             # +11 = 56 features
df = add_market_neutral_features(df)      # +10 = 66 features
df = add_interaction_features(df)         # +10 = 76 features
# Then weighted selection → 20 final features
```

### Prediction Script (`predict_nse_signals_v2.py`)

**Same Functions Added** (must match training exactly):
1. `add_interaction_features(df)` - Line 457
2. Called at Line 311 (after market-neutral features)

---

## Expected Outcomes

### Before (42 Buy signals - 2%)
- **Prediction distribution**: 98% Sell / 2% Buy
- **Mean P(Buy)**: 42.7% (suppressed by market context)
- **Top features**: 80% market context (VIX, DXY, yields)
- **Model logic**: Bearish market → Sell everything

### After (Hybrid Approach)
- **Prediction distribution**: **40-50% Buy** (balanced)
- **Mean P(Buy)**: ~50% (centered around threshold)
- **Top features**: 
  - 25% market context (VIX matters, but controlled)
  - 35% stock-specific (RSI, MACD, price action)
  - 25% relative metrics (outperformance, sector comparison)
  - 15% interactions (smart combinations)
- **Model logic**: 
  - **Bearish market + Weak stock** → Sell
  - **Bearish market + Strong stock** → Buy (contrarian opportunity)
  - **Bullish market + Weak stock** → Sell (laggard)
  - **Bullish market + Strong stock** → Buy (momentum)

---

## Why This Approach is Superior

### vs Pure Market-Neutral (User's Concern)
❌ **Problem**: Ignoring market completely is "blind mistake"
✅ **Solution**: We **keep** market context (25% of features) but **control** its influence

### vs Pure Importance-Based Selection
❌ **Problem**: Market features dominate (60-80%) due to high correlation
✅ **Solution**: Force category balance (max 35% any category)

### vs Adding More Neutral Features
❌ **Problem**: Just dilutes the pool, doesn't solve dominance
✅ **Solution**: Weighted selection ensures balanced representation

### vs Multiple Models (Market Regime-Based)
❌ **Problem**: Complex, requires regime detection, switching logic
✅ **Solution**: Single model that learns regime-appropriate responses

---

## Testing Plan

1. **Retrain Model** with hybrid approach
2. **Check Feature Distribution**: Should see ~5 market, ~7 stock, ~5 relative, ~3 interactions
3. **Run Predictions**: Expect 40-50% Buy signals (not 2% or 98%)
4. **Validate Logic**: Check if model correctly identifies:
   - Strong stocks in weak markets (Buy)
   - Weak stocks in strong markets (Sell)
5. **Monitor Over Time**: Track prediction distribution across different market regimes

---

## Key Metrics to Watch

### Feature Selection (Post-Training)
- ✅ No category > 35%
- ✅ Stock-specific ≥ 35%
- ✅ Market context ≤ 25%
- ✅ At least 2 interaction features selected

### Prediction Distribution
- ✅ Buy signals: 40-60% (not 2% or 98%)
- ✅ Mean P(Buy): 45-55% (centered)
- ✅ Standard deviation: > 0.15 (not clustered)

### Contextual Validation
- ✅ In bearish markets: Still finds 30-40% Buy candidates
- ✅ In bullish markets: Still finds 30-40% Sell candidates
- ✅ Contrarian signals work: Strong stocks + weak market = Buy

---

## Conclusion

This hybrid approach **respects the value of market context** while **preventing over-reliance**. 

It's not about ignoring VIX/DXY/yields - it's about using them **in combination with** stock-specific signals, not **instead of** them.

**Analogy**: 
- **Bad**: "Is it raining? Don't go outside." (market-only)
- **Blind**: "I never check weather." (market-neutral only)
- **Good**: "Is it raining? I have an umbrella." (hybrid) ← **THIS**

The model learns: "Market conditions matter, but stock-specific strength matters MORE."

---

## Files Modified

1. `retrain_nse_model_v2.py` - Training script with hybrid approach
2. `predict_nse_signals_v2.py` - Prediction script with same features
3. `HYBRID_APPROACH_APRIL_21_2026.md` - This documentation

---

**Next Step**: Run `python retrain_nse_model_v2.py` and validate feature distribution + prediction balance.
