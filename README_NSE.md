# NSE 500 Trading Signal Prediction System

🇮🇳 **A comprehensive machine learning system for generating trading signals for NSE 500 stocks using technical indicators and predictive analytics.**

## 🎯 Overview

This system extends the existing NASDAQ ML prediction framework to work with NSE (National Stock Exchange) 500 stocks, providing automated trading signal generation for the Indian equity market.

### 📊 Key Features

- **NSE 500 Coverage**: Supports all stocks in the NSE 500 index
- **Technical Indicators**: RSI, SMA, EMA, MACD, Bollinger Bands, ATR
- **ML Predictions**: Buy/Sell/Hold signals with confidence levels
- **Daily Automation**: Automated daily analysis and reporting
- **CSV Exports**: Ready-to-use trading data exports
- **Risk Management**: Confidence-based signal classification

## 🏗️ Database Schema

### Input Tables
- `nse_500_hist_data` - Historical price and volume data for NSE stocks
- `nse_500` - NSE 500 stock master list

### Output Tables
- `ml_nse_trading_predictions` - ML-generated trading signals
- `ml_nse_technical_indicators` - Calculated technical indicators
- `ml_nse_predict_summary` - Daily prediction summaries

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- SQL Server with NSE 500 data
- Required Python packages (see `requirements.txt`)

### Installation
1. Ensure NSE 500 data is loaded in SQL Server
2. Configure database connection in `.env` file
3. Install dependencies: `pip install -r requirements.txt`
4. Run initial setup: `python predict_nse_signals.py --check-only`

## 📈 Usage

### Single Stock Prediction
```bash
# Predict signals for a specific NSE stock
python predict_nse_signals.py --ticker RELIANCE.NS --date 2025-11-28
```

### Bulk Prediction
```bash
# Predict signals for all NSE 500 stocks
python predict_nse_signals.py --all-nse
```

### Daily Automation
```bash
# Run complete daily automation
python daily_nse_automation.py

# Windows batch script
run_nse_automation.bat
```

### Export Results
```bash
# Export all results
python export_nse_results.py --all

# Export high confidence signals only
python export_nse_results.py --high-confidence

# Create trading watchlist
python export_nse_results.py --watchlist
```

## 📊 Technical Indicators

### Momentum Indicators
- **RSI (14)**: Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **Price Momentum**: 5-day and 10-day momentum

### Trend Indicators
- **SMA**: Simple Moving Averages (5, 10, 20, 50)
- **EMA**: Exponential Moving Averages (5, 10, 20, 50)
- **Trend Direction**: Uptrend/Downtrend/Sideways

### Volatility Indicators
- **Bollinger Bands**: 20-period with 2 standard deviations
- **ATR**: Average True Range (14-period)
- **Daily Volatility**: Rolling volatility measures

### Volume Indicators
- **Volume SMA**: 20-period volume moving average
- **Volume Ratio**: Current vs average volume
- **Volume-Price Trend**: Combined volume and price momentum

## 🎯 Signal Classification

### Confidence Levels
- **High Confidence**: ≥80% model confidence
- **Medium Confidence**: 60-80% model confidence  
- **Low Confidence**: <60% model confidence

### Signal Types
- **Buy**: Bullish sentiment with positive momentum
- **Sell**: Bearish sentiment with negative momentum
- **Hold**: Neutral or uncertain market conditions

## 📁 Output Files

### CSV Exports (in `results/` directory)
- `nse_trading_predictions_YYYYMMDD_HHMMSS.csv` - All predictions
- `nse_high_confidence_signals_YYYYMMDD_HHMMSS.csv` - High confidence signals
- `nse_technical_indicators_YYYYMMDD_HHMMSS.csv` - Technical indicator values
- `nse_trading_watchlist_YYYYMMDD_HHMMSS.csv` - Focused trading watchlist

### Daily Reports (in `daily_reports/` directory)
- `nse_daily_report_YYYYMMDD.txt` - Human-readable daily report
- `nse_daily_summary_YYYYMMDD.json` - JSON summary for programmatic use

### Logs (in `logs/` directory)
- `daily_nse_automation_YYYYMMDD_HHMMSS.log` - Detailed execution logs

## 🔧 Configuration

### Database Connection
Configure your SQL Server connection in `.env` file:
```env
SQL_SERVER=192.168.86.55\MSSQLSERVER01
SQL_DATABASE=stockdata_db
SQL_USERNAME=remote_user
SQL_PASSWORD=YourStrongPassword123!
SQL_TRUSTED_CONNECTION=no
```

### Model Parameters
The system uses pre-trained models from the NASDAQ system:
- `data/best_model_extra_trees.joblib` - Extra Trees Classifier
- `data/scaler.joblib` - Feature scaler
- `data/target_encoder.joblib` - Target encoder

## 📊 Example Workflow

### Daily Automation Sequence
1. **Data Check**: Verify NSE data availability and quality
2. **Technical Analysis**: Calculate all technical indicators
3. **ML Prediction**: Generate trading signals for all stocks
4. **Result Storage**: Save predictions to database tables
5. **CSV Export**: Generate downloadable result files
6. **Report Generation**: Create daily summary reports

### Sample Output
```
🇮🇳 NSE 500 TRADING SIGNAL PREDICTION RESULTS
===============================================
📅 Analysis Date: 2025-11-28
📊 Total Predictions: 487
🟢 Buy Signals: 156
🔴 Sell Signals: 89
🟡 Hold Signals: 242
⏱️ Processing Time: 45.2 seconds

🎯 High Confidence Signals (23):
  RELIANCE.NS: Buy (87.3%) - ₹2,845.60
  TCS.NS: Hold (85.1%) - ₹3,421.80
  INFY.NS: Sell (82.4%) - ₹1,789.25
```

## 🚨 Important Notes

### Data Requirements
- NSE 500 historical data must be available in `nse_500_hist_data` table
- Minimum 60 days of historical data required for technical indicators
- Data should be updated daily for best results

### Performance Considerations
- Full NSE 500 analysis takes 30-60 seconds
- Database queries are optimized with appropriate indexes
- Large exports may take several minutes

### Risk Disclaimer
⚠️ **This system is for educational and research purposes only. Trading decisions should not be made solely based on these predictions. Always consult with financial advisors and conduct your own research before making investment decisions.**

## 🆘 Troubleshooting

### Common Issues

1. **"No NSE data found"**
   - Check if `nse_500_hist_data` table contains recent data
   - Verify ticker format (should include .NS suffix)

2. **"Model artifacts not found"**
   - Ensure model files exist in `data/` directory
   - Re-run NASDAQ training if models are missing

3. **"Database connection failed"**
   - Check `.env` configuration
   - Verify SQL Server is running and accessible

### Getting Help
- Check log files in `logs/` directory for detailed error information
- Ensure all dependencies are installed
- Verify database connectivity with `--check-only` option

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review log files for specific error messages
3. Verify NSE data availability and format
4. Test with single stock predictions before bulk processing

---

🇮🇳 **Built for the Indian equity market with NSE 500 coverage**