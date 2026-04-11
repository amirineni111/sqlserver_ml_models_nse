"""
NSE ML Prediction Configuration

Centralized configuration for confidence thresholds and model parameters.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# CONFIDENCE THRESHOLDS
# ============================================================================

# High confidence threshold (default: 60%)
# Lowered from 70% to 60% based on analysis showing:
# - NASDAQ high-confidence (>70%) = 50% accuracy (coin flip)
# - NSE 5-model calibrated ensemble is more conservative but likely more accurate
# - ~7,000 medium-confidence signals (55-70%) were being ignored daily
# - Lower threshold unlocks actionable signals while maintaining quality
HIGH_CONFIDENCE_THRESHOLD = float(os.getenv('HIGH_CONFIDENCE_THRESHOLD', '0.60'))

# Medium confidence threshold (default: 55%)
MEDIUM_CONFIDENCE_THRESHOLD = float(os.getenv('MEDIUM_CONFIDENCE_THRESHOLD', '0.55'))

# Low confidence threshold (below MEDIUM_CONFIDENCE_THRESHOLD)
# Signals below 55% are flagged as low confidence
HOLD_CONFIDENCE_THRESHOLD = float(os.getenv('HOLD_CONFIDENCE_THRESHOLD', '0.55'))
HOLD_PROBABILITY_MARGIN = float(os.getenv('HOLD_PROBABILITY_MARGIN', '0.08'))


# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Minimum prediction confidence to save to database (default: 50%)
MIN_PREDICTION_CONFIDENCE = float(os.getenv('MIN_PREDICTION_CONFIDENCE', '0.50'))

# Model accuracy threshold for retraining trigger (default: 52%)
MIN_MODEL_ACCURACY = float(os.getenv('MIN_MODEL_ACCURACY', '0.52'))

# Live prediction accuracy threshold for retraining (default: 50%)
MIN_LIVE_ACCURACY = float(os.getenv('MIN_LIVE_ACCURACY', '0.50'))


# ============================================================================
# HISTORICAL DATA PARAMETERS
# ============================================================================

# Days of historical data to fetch for predictions (default: 365)
HISTORICAL_DAYS = int(os.getenv('HISTORICAL_DAYS', '365'))

# Days of historical data for indicators calculation (default: 90)
INDICATOR_LOOKBACK_DAYS = int(os.getenv('INDICATOR_LOOKBACK_DAYS', '90'))


# ============================================================================
# VALIDATION
# ============================================================================

# Validate thresholds
if not (0.0 <= HIGH_CONFIDENCE_THRESHOLD <= 1.0):
    raise ValueError(f"HIGH_CONFIDENCE_THRESHOLD must be between 0 and 1, got {HIGH_CONFIDENCE_THRESHOLD}")

if not (0.0 <= MEDIUM_CONFIDENCE_THRESHOLD <= HIGH_CONFIDENCE_THRESHOLD):
    raise ValueError(f"MEDIUM_CONFIDENCE_THRESHOLD must be between 0 and HIGH_CONFIDENCE_THRESHOLD, got {MEDIUM_CONFIDENCE_THRESHOLD}")

if not (0.0 <= MIN_PREDICTION_CONFIDENCE <= 1.0):
    raise ValueError(f"MIN_PREDICTION_CONFIDENCE must be between 0 and 1, got {MIN_PREDICTION_CONFIDENCE}")

if not (0.0 <= HOLD_CONFIDENCE_THRESHOLD <= 1.0):
    raise ValueError(f"HOLD_CONFIDENCE_THRESHOLD must be between 0 and 1, got {HOLD_CONFIDENCE_THRESHOLD}")

if not (0.0 <= HOLD_PROBABILITY_MARGIN <= 1.0):
    raise ValueError(f"HOLD_PROBABILITY_MARGIN must be between 0 and 1, got {HOLD_PROBABILITY_MARGIN}")


# ============================================================================
# CONFIGURATION SUMMARY (for logging)
# ============================================================================

def print_config():
    """Print current configuration"""
    print("=" * 80)
    print("NSE ML PREDICTION CONFIGURATION")
    print("=" * 80)
    print(f"High Confidence Threshold:    {HIGH_CONFIDENCE_THRESHOLD:.0%} (≥{HIGH_CONFIDENCE_THRESHOLD*100:.0f}%)")
    print(f"Medium Confidence Threshold:  {MEDIUM_CONFIDENCE_THRESHOLD:.0%} (≥{MEDIUM_CONFIDENCE_THRESHOLD*100:.0f}%)")
    print(f"Low Confidence:               <{MEDIUM_CONFIDENCE_THRESHOLD:.0%} (<{MEDIUM_CONFIDENCE_THRESHOLD*100:.0f}%)")
    print(f"Hold Confidence Threshold:    {HOLD_CONFIDENCE_THRESHOLD:.0%}")
    print(f"Hold Probability Margin:      {HOLD_PROBABILITY_MARGIN:.0%}")
    print(f"Minimum Prediction Confidence: {MIN_PREDICTION_CONFIDENCE:.0%}")
    print(f"Model Accuracy Threshold:     {MIN_MODEL_ACCURACY:.0%}")
    print(f"Live Accuracy Threshold:      {MIN_LIVE_ACCURACY:.0%}")
    print(f"Historical Days:              {HISTORICAL_DAYS}")
    print(f"Indicator Lookback Days:      {INDICATOR_LOOKBACK_DAYS}")
    print("=" * 80)


if __name__ == "__main__":
    print_config()
