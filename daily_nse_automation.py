"""
Daily NSE 500 Automation Script

This script provides automated daily workflow for NSE 500 trading signals:
1. Checks NSE data availability and quality
2. Checks if model retraining is needed (weekly or on accuracy degradation)
3. Runs NSE predictions using NSE-trained ensemble models
4. Saves predictions to database
5. Monitors model performance and logs metrics
6. Creates daily reports and summaries

Usage:
    python daily_nse_automation.py                    # Full automated daily process
    python daily_nse_automation.py --check-only      # Only check data status
    python daily_nse_automation.py --date 2025-11-28 # Run for specific date
    python daily_nse_automation.py --force-retrain   # Force model retraining
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import json
import time
import pickle

# Configure UTF-8 encoding for Windows console compatibility
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import SQLServerConnection


def setup_nse_logging():
    """Setup comprehensive logging for NSE daily automation."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_filename = log_dir / f"daily_nse_automation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    class UnicodeFormatter(logging.Formatter):
        def format(self, record):
            formatted = super().format(record)
            emoji_replacements = {
                '[NSE]': '[NSE]', '[INFO]': '[INFO]', '[START]': '[START]', '[LOG]': '[LOG]',
                '[ERROR]': '[ERROR]', '[SUCCESS]': '[SUCCESS]', '[DATA]': '[DATA]', '[PROCESS]': '[PROCESS]',
                '[FILE]': '[FILES]', '[TIME]': '[TIMEOUT]', '[PREDICTION]': '[PREDICTION]', '[DATE]': '[DATE]',
                '[WARN]': '[WARN]', '[COMPLETE]': '[COMPLETE]', '[SUMMARY]': '[SUMMARY]', '[SAVE]': '[SAVE]'
            }
            for emoji, replacement in emoji_replacements.items():
                formatted = formatted.replace(emoji, replacement)
            return formatted
    
    # File handler
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(UnicodeFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return log_filename


def safe_print(text):
    """Print with safe encoding for Windows Task Scheduler compatibility."""
    text = str(text)
    
    emoji_map = {
        '\U0001f4a1': '[TIP]', '\U0001f3af': '[TARGET]', '\U0001f7e2': '[BUY]',
        '\U0001f534': '[SELL]', '\U0001f7e1': '[MEDIUM]', '\u26a1': '[FAST]', '\U0001f514': '[NOTIFY]'
    }
    
    for emoji, replacement in emoji_map.items():
        text = text.replace(emoji, replacement)
    
    try:
        print(text)
    except (UnicodeEncodeError, UnicodeDecodeError):
        safe_text = text.encode('ascii', 'ignore').decode('ascii')
        print(safe_text)


def check_nse_data_status():
    """Check NSE data availability and quality."""
    try:
        logging.info("[INFO] Checking NSE data status and connectivity...")
        
        db = SQLServerConnection()
        
        if not db.test_connection():
            logging.error("[ERROR] Database connection failed")
            return False, None, None, None
        
        logging.info("[SUCCESS] Database connection successful")
        
        # Check NSE historical data
        nse_data_query = """
        SELECT 
            COUNT(*) as total_records,
            MAX(trading_date) as latest_date,
            MIN(trading_date) as earliest_date,
            COUNT(DISTINCT ticker) as unique_tickers,
            COUNT(DISTINCT trading_date) as unique_dates
        FROM dbo.nse_500_hist_data
        """
        
        nse_result = db.execute_query(nse_data_query)
        
        if nse_result.empty:
            logging.error("[ERROR] No NSE historical data found")
            return False, None, None, None
        
        total_records = nse_result.iloc[0]['total_records']
        latest_date = nse_result.iloc[0]['latest_date']
        unique_tickers = nse_result.iloc[0]['unique_tickers']
        unique_dates = nse_result.iloc[0]['unique_dates']
        
        logging.info(f"[DATA] NSE Historical Data Status:")
        logging.info(f"  [DATE] Latest Date: {latest_date}")
        logging.info(f"  [DATA] Total Records: {total_records:,}")
        logging.info(f"  [COMPANY] Unique Tickers: {unique_tickers}")
        logging.info(f"  [DATE] Unique Dates: {unique_dates}")
        
        # Calculate data age with EST/IST timezone awareness
        # NSE closes 3:30 PM IST = ~5 AM EST, so on weekdays at 7 AM EST,
        # today's NSE data should already be in the DB (data_age = 0 days)
        if latest_date:
            if hasattr(latest_date, 'date'):
                latest_dt = latest_date.date()
            else:
                latest_dt = latest_date
            
            today = datetime.now().date()
            data_age = (today - latest_dt).days
            logging.info(f"  [TIME] Data Age: {data_age} day(s)")
            
            if data_age == 0:
                logging.info(f"  [OK] Today's NSE data available (NSE closed at ~5 AM EST)")
            elif data_age <= 2 and today.weekday() in [0]:  # Monday
                logging.info(f"  [OK] Monday - latest data from Friday ({latest_dt}) is expected")
            elif data_age <= 2 and today.weekday() in [5, 6]:  # Weekend
                logging.info(f"  [OK] Weekend - latest data from Friday ({latest_dt}) is expected")
            elif data_age > 2 and today.weekday() not in [5, 6, 0]:
                logging.warning(f"  [WARN] Data is {data_age} days old on a weekday - "
                               f"check NSE data feed!")
        else:
            data_age = None
        
        # Check for recent predictions
        pred_query = """
        SELECT 
            COUNT(*) as prediction_count,
            MAX(trading_date) as latest_prediction_date
        FROM ml_nse_trading_predictions
        WHERE trading_date >= DATEADD(day, -7, GETDATE())
        """
        
        pred_result = db.execute_query(pred_query)
        prediction_count = pred_result.iloc[0]['prediction_count'] if not pred_result.empty else 0
        latest_prediction_date = pred_result.iloc[0]['latest_prediction_date'] if not pred_result.empty else None
        
        logging.info(f"[PREDICTION] Recent NSE Predictions:")
        logging.info(f"  [DATA] Last 7 Days: {prediction_count} predictions")
        logging.info(f"  [DATE] Latest Prediction: {latest_prediction_date}")
        
        return True, data_age, latest_date, total_records
        
    except Exception as e:
        logging.error(f"[ERROR] Error checking NSE data status: {e}")
        return False, None, None, None


def check_model_status():
    """Check NSE model status and determine if retraining is needed."""
    logging.info("[INFO] Checking NSE model status...")
    
    nse_model_dir = Path('data/nse_models')
    metadata_path = nse_model_dir / 'nse_training_metadata.pkl'
    
    needs_retrain = False
    reason = ""
    
    # Check if NSE models exist
    if not metadata_path.exists():
        logging.warning("=" * 60)
        logging.warning("[WARN] NO NSE-SPECIFIC MODELS FOUND!")
        logging.warning("[WARN] The system will fall back to legacy NASDAQ-trained models")
        logging.warning("[WARN] which produce UNRELIABLE predictions for NSE stocks.")
        logging.warning("[WARN] NSE retraining will be triggered now.")
        logging.warning("=" * 60)
        needs_retrain = True
        reason = "No NSE-specific models found - using NASDAQ models is unreliable"
        return needs_retrain, reason
    
    # Load metadata
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        training_timestamp = metadata.get('training_timestamp', '')
        best_clf = metadata.get('best_clf_model', 'Unknown')
        clf_results = metadata.get('clf_results', {})
        reg_results = metadata.get('reg_results', {})
        
        logging.info(f"  [OK] Models found - trained at: {training_timestamp}")
        logging.info(f"  [OK] Best classifier: {best_clf}")
        
        # Check model age
        if training_timestamp:
            try:
                train_date = datetime.strptime(training_timestamp[:8], '%Y%m%d')
                model_age_days = (datetime.now() - train_date).days
                logging.info(f"  [TIME] Model age: {model_age_days} days")
                
                if model_age_days > 7:
                    needs_retrain = True
                    reason = f"Model is {model_age_days} days old (>7 days)"
                    logging.warning(f"  [WARN] {reason}")
            except Exception:
                pass
        
        # Check model accuracy from metadata
        if best_clf in clf_results:
            accuracy = clf_results[best_clf].get('accuracy', 0)
            f1 = clf_results[best_clf].get('f1_score', 0)
            logging.info(f"  [DATA] Best model accuracy: {accuracy:.1%}, F1: {f1:.3f}")
            
            if accuracy < 0.52:
                needs_retrain = True
                reason = f"Model accuracy ({accuracy:.1%}) below threshold (52%)"
                logging.warning(f"  [WARN] {reason}")
        
        # Check recent prediction accuracy from ai_prediction_history
        try:
            db = SQLServerConnection()
            accuracy_query = """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) as correct,
                CAST(SUM(CASE WHEN direction_correct = 1 THEN 1.0 ELSE 0.0 END) / 
                     NULLIF(COUNT(*), 0) * 100 AS DECIMAL(5,2)) as accuracy_pct
            FROM dbo.ai_prediction_history 
            WHERE market = 'NSE 500'
                AND target_date >= DATEADD(day, -7, CAST(GETDATE() AS DATE))
                AND actual_price IS NOT NULL
            """
            accuracy_result = db.execute_query(accuracy_query)
            
            if not accuracy_result.empty and accuracy_result.iloc[0]['total'] > 0:
                live_accuracy = float(accuracy_result.iloc[0]['accuracy_pct'])
                total = accuracy_result.iloc[0]['total']
                logging.info(f"  [DATA] Live prediction accuracy (7 days): {live_accuracy:.1f}% ({total} predictions)")
                
                if live_accuracy < 50.0 and total > 100:
                    needs_retrain = True
                    reason = f"Live accuracy ({live_accuracy:.1f}%) below 50%"
                    logging.warning(f"  [WARN] {reason}")
        except Exception as e:
            logging.warning(f"  [WARN] Could not check live accuracy: {e}")
        
    except Exception as e:
        logging.error(f"  [ERROR] Error reading model metadata: {e}")
        needs_retrain = True
        reason = f"Error reading metadata: {e}"
    
    if not needs_retrain:
        logging.info("  [SUCCESS] Models are current and performing adequately")
    
    return needs_retrain, reason


def run_model_retrain():
    """Run NSE model retraining."""
    try:
        logging.info("[START] Running NSE model retraining...")
        
        cmd_args = [sys.executable, "retrain_nse_model.py", "--quick", "--backup-old"]
        
        logging.info(f"[PROCESS] Executing: {' '.join(cmd_args)}")
        
        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            timeout=3600,  # 60 minutes timeout for retraining
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0:
            logging.info("[SUCCESS] NSE model retraining completed successfully")
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-15:]:
                if line.strip():
                    logging.info(f"  {line}")
            return True
        else:
            logging.error(f"[ERROR] NSE model retraining failed with return code: {result.returncode}")
            if result.stderr:
                logging.error(f"Error output: {result.stderr[-500:]}")
            if result.stdout:
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines[-10:]:
                    logging.error(f"  {line}")
            return False
        
    except subprocess.TimeoutExpired:
        logging.error("[TIME] NSE model retraining timed out after 60 minutes")
        return False
    except Exception as e:
        logging.error(f"[ERROR] Error running NSE model retraining: {e}")
        return False


def run_nse_predictions(target_date=None):
    """Run NSE trading signal predictions."""
    try:
        logging.info("[START] Running NSE 500 predictions...")
        
        cmd_args = [sys.executable, "predict_nse_signals.py", "--all-nse"]
        
        if target_date:
            cmd_args.extend(["--date", target_date])
        
        logging.info(f"[PROCESS] Executing: {' '.join(cmd_args)}")
        
        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minutes timeout
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0:
            logging.info("[SUCCESS] NSE predictions completed successfully")
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-10:]:
                if line.strip():
                    logging.info(f"[DATA] {line}")
            return True
        else:
            logging.error(f"[ERROR] NSE predictions failed with return code: {result.returncode}")
            if result.stderr:
                logging.error(f"Error output: {result.stderr[-500:]}")
            return False
        
    except subprocess.TimeoutExpired:
        logging.error("[TIME] NSE predictions timed out after 30 minutes")
        return False
    except Exception as e:
        logging.error(f"[ERROR] Error running NSE predictions: {e}")
        return False


def log_model_performance():
    """Log current model performance metrics."""
    try:
        logging.info("[DATA] Checking model performance metrics...")
        
        db = SQLServerConnection()
        
        # 7-day accuracy by model
        accuracy_query = """
        SELECT 
            model_name,
            COUNT(*) as total,
            SUM(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) as correct,
            CAST(SUM(CASE WHEN direction_correct = 1 THEN 1.0 ELSE 0.0 END) / 
                 NULLIF(COUNT(*), 0) * 100 AS DECIMAL(5,2)) as accuracy_pct,
            AVG(CAST(percentage_error AS FLOAT)) as avg_pct_error
        FROM dbo.ai_prediction_history 
        WHERE market = 'NSE 500'
            AND target_date >= DATEADD(day, -7, CAST(GETDATE() AS DATE))
            AND actual_price IS NOT NULL
        GROUP BY model_name
        ORDER BY accuracy_pct DESC
        """
        
        accuracy_df = db.execute_query(accuracy_query)
        
        if not accuracy_df.empty:
            logging.info("  [DATA] NSE 500 Model Performance (Last 7 Days):")
            logging.info(f"  {'Model':<25} {'Total':<10} {'Correct':<10} {'Accuracy':<12} {'Avg Error'}")
            logging.info(f"  {'-'*70}")
            
            for _, row in accuracy_df.iterrows():
                logging.info(
                    f"  {row['model_name']:<25} {row['total']:<10} {row['correct']:<10} "
                    f"{row['accuracy_pct']:.1f}%{' ':>6} {row['avg_pct_error']:.2f}%"
                )
            
            # Alert if all models below 50%
            max_accuracy = accuracy_df['accuracy_pct'].max()
            if max_accuracy < 50:
                logging.warning(f"  [WARN] ALL models below 50% accuracy! Best: {max_accuracy:.1f}%")
                logging.warning(f"  [WARN] Consider running: python retrain_nse_model.py")
        else:
            logging.info("  [INFO] No recent prediction results to evaluate")
        
        # Today's prediction summary
        summary_query = """
        SELECT TOP 1
            analysis_date,
            total_predictions,
            total_buy_signals,
            total_sell_signals,
            high_confidence_count,
            avg_confidence,
            market_trend,
            model_accuracy
        FROM ml_nse_predict_summary
        ORDER BY analysis_date DESC
        """
        
        summary_df = db.execute_query(summary_query)
        
        if not summary_df.empty:
            row = summary_df.iloc[0]
            logging.info(f"\n  [DATA] Latest Prediction Summary ({row['analysis_date']}):")
            logging.info(f"    Total Predictions: {row['total_predictions']}")
            logging.info(f"    Buy/Sell: {row['total_buy_signals']}/{row['total_sell_signals']}")
            logging.info(f"    High Confidence: {row['high_confidence_count']}")
            logging.info(f"    Avg Confidence: {row['avg_confidence']:.2f}")
            logging.info(f"    Market Trend: {row['market_trend']}")
            if row.get('model_accuracy'):
                logging.info(f"    Model Accuracy: {row['model_accuracy']:.1%}")
        
    except Exception as e:
        logging.error(f"[ERROR] Error checking model performance: {e}")


def generate_daily_report(target_date=None):
    """Generate daily NSE analysis report."""
    try:
        logging.info("[LOG] Generating daily NSE report...")
        
        db = SQLServerConnection()
        
        # Get the latest analysis date from the summary table
        # This handles the NSE timing: analysis_date = latest trading date from data,
        # NOT necessarily today (could be Friday's data on a Saturday/Sunday)
        if target_date:
            date_filter = f"analysis_date = '{target_date}'"
        else:
            # Get the most recent summary regardless of date
            date_filter = "1=1"
        
        summary_query = f"""
        SELECT TOP 1
            analysis_date,
            total_predictions,
            total_buy_signals,
            total_sell_signals,
            total_hold_signals,
            high_confidence_count,
            medium_confidence_count,
            low_confidence_count,
            avg_confidence,
            avg_rsi,
            total_stocks_processed,
            processing_time_seconds,
            market_trend,
            model_accuracy,
            run_timestamp
        FROM ml_nse_predict_summary
        WHERE {date_filter}
        ORDER BY created_at DESC
        """
        
        summary_result = db.execute_query(summary_query)
        
        # Determine the actual trading date from the summary
        if not summary_result.empty:
            actual_trading_date = str(summary_result.iloc[0]['analysis_date'])
            if hasattr(summary_result.iloc[0]['analysis_date'], 'strftime'):
                actual_trading_date = summary_result.iloc[0]['analysis_date'].strftime('%Y-%m-%d')
        else:
            actual_trading_date = target_date or datetime.now().strftime('%Y-%m-%d')
        
        # Get high confidence signals using the actual trading date
        high_conf_query = f"""
        SELECT TOP 10
            ticker,
            company,
            predicted_signal,
            confidence_percentage,
            close_price,
            rsi,
            model_name
        FROM ml_nse_trading_predictions
        WHERE trading_date = '{actual_trading_date}'
            AND high_confidence = 1
        ORDER BY confidence_percentage DESC
        """
        
        high_conf_result = db.execute_query(high_conf_query)
        
        # Use the actual trading date for report filename (not today's date)
        report_date = actual_trading_date
        # Determine run context
        run_date = datetime.now().strftime('%Y-%m-%d')
        nse_model_exists = Path('data/nse_models/nse_training_metadata.pkl').exists()
        model_type = 'NSE-trained ensemble' if nse_model_exists else 'Legacy (NASDAQ-trained)'
        
        report_content = f"""
NSE 500 Daily Trading Report
================================
[DATE] NSE Trading Date (data from): {report_date}
[TIME] Report Generated (EST): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
[MODEL] Model Type: {model_type}
[NOTE] Predictions are for the NEXT trading day after {report_date}
[NOTE] NSE closes 3:30 PM IST (~5:00 AM EST) - data available before this report runs

"""
        
        if not summary_result.empty:
            row = summary_result.iloc[0]
            report_content += f"""
[DATA] PREDICTION SUMMARY
--------------------
[PREDICTION] Total Predictions: {row['total_predictions']}
[BUY] Buy Signals: {row['total_buy_signals']}
[SELL] Sell Signals: {row['total_sell_signals']}
[HOLD] Hold Signals: {row.get('total_hold_signals', 0)}
[TREND] Market Trend: {row.get('market_trend', 'Unknown')}

[TARGET] CONFIDENCE BREAKDOWN
-----------------------
[HIGH] High Confidence: {row['high_confidence_count']}
[DATA] Medium Confidence: {row['medium_confidence_count']}
[LOW] Low Confidence: {row['low_confidence_count']}

[PREDICTION] ANALYSIS METRICS
-------------------
[STAR] Average Confidence: {row['avg_confidence']:.2f}
[DATA] Average RSI: {row.get('avg_rsi', 0):.1f}
[COMPANY] Stocks Processed: {row['total_stocks_processed']}
"""
            if row.get('model_accuracy'):
                report_content += f"[MODEL] Model Accuracy: {row['model_accuracy']:.1%}\n"
            if row.get('processing_time_seconds'):
                report_content += f"[TIME] Processing Time: {row['processing_time_seconds']:.1f} seconds\n"
        
        if not high_conf_result.empty:
            report_content += """
[TARGET] TOP HIGH CONFIDENCE SIGNALS
-------------------------------
"""
            for _, row in high_conf_result.iterrows():
                signal_tag = "[BUY]" if row['predicted_signal'] == 'Buy' else "[SELL]" if row['predicted_signal'] == 'Sell' else "[HOLD]"
                model_info = f" [{row.get('model_name', '')}]" if row.get('model_name') else ""
                report_content += (
                    f"{signal_tag} {row['ticker']}: {row['predicted_signal']} "
                    f"({row['confidence_percentage']:.1f}%) - INR {row['close_price']:.2f}{model_info}\n"
                )
        
        # Save report
        reports_dir = Path("daily_reports")
        reports_dir.mkdir(exist_ok=True)
        
        report_filename = reports_dir / f"nse_daily_report_{report_date.replace('-', '')}.txt"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # JSON summary
        json_report = {
            'date': report_date,
            'generated_at': datetime.now().isoformat(),
            'summary': summary_result.to_dict('records')[0] if not summary_result.empty else {},
            'high_confidence_signals': high_conf_result.to_dict('records') if not high_conf_result.empty else [],
            'model_type': 'NSE-trained ensemble' if Path('data/nse_models/nse_training_metadata.pkl').exists() else 'Legacy',
        }
        
        json_filename = reports_dir / f"nse_daily_summary_{report_date.replace('-', '')}.json"
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, default=str)
        
        logging.info(f"[LOG] Daily report saved: {report_filename}")
        logging.info(f"[DATA] JSON summary saved: {json_filename}")
        
        return True
        
    except Exception as e:
        logging.error(f"[ERROR] Error generating daily report: {e}")
        return False


def main():
    """Main automation function."""
    parser = argparse.ArgumentParser(description='Daily NSE 500 Automation')
    parser.add_argument('--check-only', action='store_true', help='Only check data status')
    parser.add_argument('--date', help='Target date for analysis (YYYY-MM-DD)')
    parser.add_argument('--skip-predictions', action='store_true', help='Skip prediction generation')
    parser.add_argument('--force-retrain', action='store_true', help='Force model retraining')
    parser.add_argument('--skip-retrain-check', action='store_true', help='Skip retraining check')
    
    args = parser.parse_args()
    
    # Setup logging
    log_filename = setup_nse_logging()
    
    start_time = datetime.now()
    
    safe_print("[START] Starting Daily NSE 500 Automation")
    safe_print(f"[DATE] Target Date: {args.date or 'Today (EST)'}")
    safe_print(f"[TIME] Run Time (EST): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print(f"[INFO] Pipeline: yfinance fetch at 7 AM EST -> ML predictions at 7:30 AM EST")
    safe_print(f"[INFO] NSE closes 3:30 PM IST (~5 AM EST) -> data fetched at 7 AM -> ready now")
    safe_print(f"[LOG] Log File: {log_filename}")
    
    success_count = 0
    total_steps = 0
    
    try:
        # Step 1: Check data status
        total_steps += 1
        logging.info("=" * 60)
        logging.info("[INFO] STEP 1: Checking NSE Data Status")
        logging.info("=" * 60)
        
        is_connected, data_age, latest_date, total_records = check_nse_data_status()
        
        if not is_connected:
            logging.error("[ERROR] Cannot proceed without database connectivity")
            return
        
        success_count += 1
        
        if args.check_only:
            logging.info("[SUCCESS] Data status check completed")
            log_model_performance()
            return
        
        # Step 2: Check model status and retrain if needed
        if not args.skip_retrain_check:
            total_steps += 1
            logging.info("=" * 60)
            logging.info("[INFO] STEP 2: Checking Model Status")
            logging.info("=" * 60)
            
            needs_retrain, reason = check_model_status()
            
            if args.force_retrain or needs_retrain:
                if args.force_retrain:
                    logging.info("[INFO] Forced retraining requested")
                else:
                    logging.info(f"[INFO] Retraining needed: {reason}")
                
                retrain_success = run_model_retrain()
                if retrain_success:
                    success_count += 1
                    logging.info("[SUCCESS] Model retraining completed")
                else:
                    logging.warning("[WARN] Model retraining failed, continuing with existing models")
                    success_count += 1  # Don't block predictions
            else:
                logging.info("[SUCCESS] Models are up to date")
                success_count += 1
        
        # Step 3: Run predictions
        if not args.skip_predictions:
            total_steps += 1
            logging.info("=" * 60)
            logging.info("[PREDICTION] STEP 3: Running NSE Predictions")
            logging.info("=" * 60)
            
            prediction_success = run_nse_predictions(args.date)
            if prediction_success:
                success_count += 1
                logging.info("[SUCCESS] NSE predictions completed successfully")
            else:
                logging.error("[ERROR] NSE predictions failed")
        
        # Step 4: Log model performance
        total_steps += 1
        logging.info("=" * 60)
        logging.info("[DATA] STEP 4: Model Performance Monitoring")
        logging.info("=" * 60)
        
        log_model_performance()
        success_count += 1
        
        # Step 5: Generate daily report
        total_steps += 1
        logging.info("=" * 60)
        logging.info("[LOG] STEP 5: Generating Daily Report")
        logging.info("=" * 60)
        
        report_success = generate_daily_report(args.date)
        if report_success:
            success_count += 1
            logging.info("[SUCCESS] Daily report generated successfully")
        else:
            logging.error("[ERROR] Daily report generation failed")
        
        # Final summary
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        logging.info("=" * 60)
        logging.info("[COMPLETE] NSE AUTOMATION SUMMARY")
        logging.info("=" * 60)
        logging.info(f"[TIME] Total Processing Time: {processing_time:.1f} seconds")
        logging.info(f"[SUCCESS] Successful Steps: {success_count}/{total_steps}")
        logging.info(f"[DATE] Date Processed: {args.date or datetime.now().strftime('%Y-%m-%d')}")
        
        # Check if using NSE models
        nse_model_exists = Path('data/nse_models/nse_training_metadata.pkl').exists()
        logging.info(f"[MODEL] Using: {'NSE-trained ensemble models' if nse_model_exists else 'Legacy models (run retrain_nse_model.py)'}")
        
        if success_count == total_steps:
            safe_print("[COMPLETE] NSE automation completed successfully!")
            logging.info("[COMPLETE] All automation steps completed successfully")
        else:
            safe_print(f"[WARN] NSE automation completed with {total_steps - success_count} failures")
            logging.warning(f"Automation completed with {total_steps - success_count} failed steps")
    
    except KeyboardInterrupt:
        logging.info("[STOP] NSE automation interrupted by user")
        safe_print("[STOP] Process interrupted by user")
    except Exception as e:
        logging.error(f"[ERROR] Fatal error in NSE automation: {e}")
        safe_print(f"[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
